import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.core.auth import resolve_request_context
from app.core.ensemble import ensemble_enabled, execute_ensemble
from app.core.exceptions import (
    ModelRouterError,
    ModelAccessError,
    ProviderError,
    AllKeysFailedError,
)
from app.core.routing_policy import DEFAULT_MODEL_ALIASES, load_routing_policy
from app.core.request_context import RequestContext
from app.core.routing_settings import load_effective_routing_settings
from app.schemas.chat import (
    ChatRequest,
    Usage,
    ChatResponse,
    CascadeMetadata,
    ChatChoice,
    ChatMessage,
    RoutingRetryMetadata,
)
from app.core.logging import log_request
from rsl.collector import build_outcome, record_outcome_safely

router = APIRouter()
logger = logging.getLogger("model_router.chat")


def _calc_cost(pricing, pt: int, ct: int) -> float:
    if not pricing:
        return 0.0
    return (pt / 1_000_000) * pricing.prompt + (ct / 1_000_000) * pricing.completion


def _model_unavailable_retry_limit() -> int:
    raw = __import__("os").environ.get("ROUTER_MODEL_UNAVAILABLE_RETRIES", "3").strip()
    try:
        return max(0, min(10, int(raw)))
    except ValueError:
        return 3


def _baseline_access_allowed(
    routing_policy,
    request_context: RequestContext,
    router_engine,
    requested_target_model: str,
    baseline_model: str,
) -> bool:
    # Mirrors the original access checks so RSL only engages on requests that
    # would already have been permitted to run the baseline model.
    if requested_target_model in router_engine.model_map and not routing_policy.is_model_allowed(
        requested_target_model,
        request_context.allowed_models,
    ):
        return False
    return routing_policy.is_model_allowed(baseline_model, request_context.allowed_models)


async def _write_log(
    db,
    body,
    provider_name,
    model_config,
    request_context: RequestContext,
    elapsed,
    usage=None,
    error=None,
    cost_override: float | None = None,
    model_override: str | None = None,
):
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    cost = cost_override if cost_override is not None else (
        _calc_cost(model_config.pricing, pt, ct) if model_config and model_config.pricing else 0.0
    )
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **request_context.log_fields(),
        "model": model_override or body.model,
        "provider": provider_name,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "cost": round(cost, 6),
        "latency_ms": elapsed,
        "status": "error" if error else "success",
        "error": error,
    }
    await db.insert_usage(row)
    asyncio.create_task(asyncio.to_thread(log_request, row))


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    request_context = await resolve_request_context(request)
    db = request.app.state.db
    routing_policy = await load_routing_policy(db, request_context.routing_profile_id)
    routing_settings = await load_effective_routing_settings(
        db,
        routing_profile_id=request_context.routing_profile_id,
        keyword_default=getattr(request.app.state, "keyword_routing_enabled", False),
        semantic_default=getattr(request.app.state, "semantic_routing_enabled", False),
        cascade_default=getattr(request.app.state, "cascade_enabled", False),
    )
    router_engine = request.app.state.router
    cascade_engine = request.app.state.cascade
    original_model = body.model
    requested_target_model = original_model
    intent_category = None
    ensemble_requested = ensemble_enabled(body)
    available_models = set(router_engine.model_map.keys())
    allowed_models = routing_policy.effective_allowed_models(
        request_context.allowed_models,
        available_models=available_models,
    )
    recommendation_applied = False
    rule_route_applied = False

    if not ensemble_requested:
        requested_is_auto = original_model in DEFAULT_MODEL_ALIASES
        if not requested_is_auto:
            requested_target_model = router_engine.resolve_model_target(
                original_model,
                default_model=routing_policy.default_model,
                preferred_providers=routing_policy.preferred_providers,
                cost_bias=routing_policy.cost_bias_value,
                latency_bias=routing_policy.latency_bias_value,
                quality_bias=routing_policy.quality_bias_value,
            )

        recommender = getattr(request.app.state, "recommender", None)
        recommendation_requested = requested_is_auto or routing_policy.recommendation_apply_to_explicit
        if routing_policy.recommendation_enabled and recommendation_requested and recommender:
            recommendation = recommender.recommend(
                messages=body.messages,
                router_engine=router_engine,
                policy=routing_policy,
                allowed_models=allowed_models,
            )
            if recommendation.applied and recommendation.selected_model:
                body.model = recommendation.selected_model
                recommendation_applied = True
            else:
                if requested_is_auto:
                    requested_target_model = router_engine.resolve_model_target(
                        original_model,
                        default_model=routing_policy.default_model,
                        preferred_providers=routing_policy.preferred_providers,
                        cost_bias=routing_policy.cost_bias_value,
                        latency_bias=routing_policy.latency_bias_value,
                        quality_bias=routing_policy.quality_bias_value,
                    )
                body.model = requested_target_model
        else:
            if requested_is_auto:
                requested_target_model = router_engine.resolve_model_target(
                    original_model,
                    default_model=routing_policy.default_model,
                    preferred_providers=routing_policy.preferred_providers,
                    cost_bias=routing_policy.cost_bias_value,
                    latency_bias=routing_policy.latency_bias_value,
                    quality_bias=routing_policy.quality_bias_value,
                )
            body.model = requested_target_model

    if not ensemble_requested and not recommendation_applied:
        keyword_enabled = routing_settings["keyword_routing_enabled"]
        if keyword_enabled:
            matched_key = await router_engine.match_keyword(
                body.messages,
                routing_profile_id=request_context.routing_profile_id,
            )
            if matched_key and matched_key in router_engine.model_map:
                body.model = matched_key
                rule_route_applied = True

        if body.model == original_model:
            semantic_enabled = routing_settings["semantic_routing_enabled"]
            if semantic_enabled:
                embedder = getattr(request.app.state, "embedder", None)
                result = await router_engine.match_semantic(body.messages, embedder)
                if result:
                    target_key, category = result
                    body.model = target_key
                    intent_category = category
                    rule_route_applied = True

    cascade_enabled_global = routing_settings["cascade_enabled"]
    cascade_config = None
    if not ensemble_requested:
        if body.cascade:
            cascade_config = body.cascade.model_dump()
        elif cascade_enabled_global:
            cascade_config = {"enabled": True}
        if routing_policy.cascade_override_chain:
            cascade_config = {
                **(cascade_config or {"enabled": True}),
                "chain": list(routing_policy.cascade_override_chain),
            }

    cascade_active = bool(cascade_config and cascade_config.get("enabled") is not False)
    cascade_source_model = body.model
    cascade_chain = None
    if cascade_active:
        cascade_chain = await cascade_engine.resolve_chain(
            cascade_source_model,
            cascade_config,
            routing_profile_id=request_context.routing_profile_id,
        )
        if cascade_chain and allowed_models is not None:
            cascade_chain = [entry for entry in cascade_chain if entry[0] in allowed_models]

    # RSL decision stage: runs after the original router produced the baseline and
    # before the original access check, which still validates the final model.
    rsl_engine = getattr(request.app.state, "rsl_engine", None)
    rsl_store = getattr(request.app.state, "rsl_store", None)
    rsl_decision = None
    rsl_request_id = None
    decision_body = body
    candidate_models = None
    decision_baseline_model = body.model
    rsl_replace_allowed = (
        original_model in DEFAULT_MODEL_ALIASES and not rule_route_applied
    )
    rsl_mode_supported = True
    if ensemble_requested:
        decision_baseline_model = (
            (body.ensemble.judge_model or "").strip()
            if body.ensemble and body.ensemble.judge_model
            else ""
        )
        rsl_mode_supported = (
            decision_baseline_model in router_engine.model_map
            and not body.stream
        )
        rsl_replace_allowed = True
    elif cascade_active:
        rsl_mode_supported = bool(cascade_chain) and not body.stream
        if cascade_chain:
            decision_baseline_model = cascade_chain[0][0]
            candidate_models = {entry[0] for entry in cascade_chain}
            rsl_replace_allowed = True
    if decision_baseline_model:
        decision_body = body.model_copy(update={"model": decision_baseline_model})

    baseline_is_allowed = (
        routing_policy.is_model_allowed(
            decision_baseline_model,
            request_context.allowed_models,
        )
        if ensemble_requested
        else _baseline_access_allowed(
            routing_policy,
            request_context,
            router_engine,
            requested_target_model,
            decision_baseline_model,
        )
    )
    if (
        rsl_engine is not None
        and rsl_store is not None
        and getattr(request.app.state, "rsl_recording_enabled", False)
        and rsl_mode_supported
        and baseline_is_allowed
    ):
        try:
            rsl_decision = await rsl_engine.decide(
                request_context=request_context,
                body=decision_body,
                routing_policy=routing_policy,
                requested_model=original_model,
                replace_allowed=rsl_replace_allowed,
                candidate_models=candidate_models,
                recommender=getattr(request.app.state, "recommender", None),
            )
        except Exception:
            logger.exception("RSL decision failed; keeping baseline model")
            rsl_decision = None
        if rsl_decision is not None:
            rsl_request_id = rsl_decision.request_id if rsl_decision.recorded else None
            if rsl_decision.applied:
                if ensemble_requested and body.ensemble:
                    body.ensemble.judge_model = rsl_decision.selected_model
                else:
                    body.model = rsl_decision.selected_model
                    if cascade_active and cascade_chain:
                        selected_entry = next(
                            (
                                entry for entry in cascade_chain
                                if entry[0] == rsl_decision.selected_model
                            ),
                            None,
                        )
                        if selected_entry:
                            cascade_chain = [
                                selected_entry,
                                *(
                                    entry for entry in cascade_chain
                                    if entry[0] != rsl_decision.selected_model
                                ),
                            ]

    if not ensemble_requested and requested_target_model in router_engine.model_map and not routing_policy.is_model_allowed(
        requested_target_model,
        request_context.allowed_models,
    ):
        raise ModelAccessError(requested_target_model)
    if not ensemble_requested and not routing_policy.is_model_allowed(
        body.model,
        request_context.allowed_models,
    ):
        raise ModelAccessError(body.model)

    rsl_fallback_model = (
        rsl_decision.baseline_model
        if rsl_decision is not None and rsl_decision.applied
        else None
    )
    rsl_fallback_reason = None

    def _rsl_outcome(
        response,
        model_config,
        status,
        latency_ms,
        error=None,
        executed_model=None,
        actual_cost_override=None,
        override_actual_cost=False,
    ):
        return build_outcome(
            request_id=rsl_request_id,
            response=response,
            body=body,
            model_config=model_config,
            status=status,
            latency_ms=latency_ms,
            error=error,
            fallback_reason=rsl_fallback_reason,
            executed_model=executed_model,
            actual_cost_override=actual_cost_override,
            override_actual_cost=override_actual_cost,
        )

    def _try_rsl_fallback(error, attempted_model, provider_label, model_config) -> bool:
        nonlocal rsl_fallback_model, rsl_fallback_reason
        if not rsl_fallback_model or attempted_model == rsl_fallback_model:
            return False
        rsl_fallback_reason = type(error).__name__
        rsl_decision.fallback_reason = rsl_fallback_reason
        rsl_decision.details["fallback_from_model"] = attempted_model
        logger.warning(
            "RSL selected model %s failed (%s); falling back to baseline %s",
            attempted_model,
            rsl_fallback_reason,
            rsl_fallback_model,
        )
        background_tasks.add_task(
            _write_log,
            db,
            body,
            provider_label,
            model_config,
            request_context,
            elapsed(),
            error=str(error),
            model_override=attempted_model,
        )
        body.model = rsl_fallback_model
        rsl_fallback_model = None
        return True

    def _record_ensemble_fallback(reason: str, attempted_model: str) -> None:
        nonlocal rsl_fallback_reason
        rsl_fallback_reason = reason
        if rsl_decision is not None:
            rsl_decision.fallback_reason = reason
            rsl_decision.details["fallback_from_model"] = attempted_model
        if body.ensemble and rsl_fallback_model:
            body.ensemble.judge_model = rsl_fallback_model

    start = time.monotonic()
    elapsed = lambda: int((time.monotonic() - start) * 1000)

    if ensemble_requested:
        try:
            resp, total_cost, judge_model_config, ensemble_fallback_reason = await execute_ensemble(
                body,
                router_engine,
                allowed_models=allowed_models,
                judge_fallback_model=rsl_fallback_model,
                judge_fallback_callback=_record_ensemble_fallback,
            )
            if ensemble_fallback_reason and rsl_decision is not None:
                rsl_fallback_reason = ensemble_fallback_reason
                rsl_decision.fallback_reason = ensemble_fallback_reason
                rsl_decision.details["fallback_from_model"] = rsl_decision.selected_model
            executed_judge = (
                resp.ensemble.final_model
                if resp.ensemble
                else body.ensemble.judge_model
            )
            if rsl_decision is not None and rsl_decision.recorded:
                resp.rsl = rsl_decision.response_metadata(executed_model=executed_judge)
            ms = elapsed()
            background_tasks.add_task(
                _write_log,
                db,
                body,
                "Ensemble",
                judge_model_config,
                request_context,
                ms,
                resp.usage,
                None,
                total_cost,
                original_model,
            )
            if rsl_request_id:
                await record_outcome_safely(
                    rsl_store,
                    _rsl_outcome(
                        resp,
                        judge_model_config,
                        "success",
                        ms,
                        executed_model=executed_judge,
                        actual_cost_override=total_cost,
                        override_actual_cost=True,
                    ),
                )
            return resp
        except ModelRouterError as e:
            if rsl_request_id:
                judge_key = body.ensemble.judge_model if body.ensemble else body.model
                judge_config = router_engine.model_map.get(judge_key, (None, None))[1]
                await record_outcome_safely(
                    rsl_store,
                    _rsl_outcome(
                        None,
                        judge_config,
                        "error",
                        elapsed(),
                        error=e,
                        executed_model=judge_key,
                    ),
                )
            raise
        except Exception as e:
            ms = elapsed()
            if rsl_request_id:
                judge_key = body.ensemble.judge_model if body.ensemble else body.model
                judge_config = router_engine.model_map.get(judge_key, (None, None))[1]
                await record_outcome_safely(
                    rsl_store,
                    _rsl_outcome(
                        None,
                        judge_config,
                        "error",
                        ms,
                        error=e,
                        executed_model=judge_key,
                    ),
                )
            background_tasks.add_task(
                _write_log,
                db,
                body,
                "Ensemble",
                None,
                request_context,
                ms,
                None,
                str(e),
                None,
                original_model,
            )
            raise

    if cascade_config and cascade_config.get("enabled") is not False:
        chain = cascade_chain
        if chain:
            validator = cascade_engine.get_validator(
                cascade_source_model,
                cascade_config,
                routing_profile_id=request_context.routing_profile_id,
            )
            local_judge = cascade_engine.get_local_judge(
                cascade_source_model,
                cascade_config,
                routing_profile_id=request_context.routing_profile_id,
                allowed_models=allowed_models,
            )
            webhook_validator = cascade_engine.get_webhook_validator(
                cascade_source_model,
                routing_profile_id=request_context.routing_profile_id,
            )
            max_escalations = cascade_engine.get_max_escalations(
                cascade_source_model,
                cascade_config,
                routing_profile_id=request_context.routing_profile_id,
            )

            if body.stream:
                return StreamingResponse(
                    _stream_cascade(
                        body, chain, validator, local_judge, webhook_validator, max_escalations,
                        db, cascade_engine, request_context, elapsed, original_model, intent_category,
                    ),
                    media_type="text/event-stream",
                )

            cascade_error = None
            try:
                resp, cascade_meta = await cascade_engine.execute_cascade(
                    body, original_model, chain, validator, local_judge, webhook_validator,
                    max_escalations, request_context.api_key_id,
                )
                ms = elapsed()
                final_model_config = next(
                    (mc for rk, _, mc in chain if rk == cascade_meta.final_model),
                    chain[-1][2],
                )

                if resp and resp.usage:
                    background_tasks.add_task(
                        _write_log, db, body, "Cascade",
                        final_model_config, request_context, ms, resp.usage,
                    )
                elif resp is None:
                    cascade_error = ProviderError(
                        "cascade",
                        502,
                        f"All cascade models failed: {cascade_meta.escalation_reason or 'no response'}",
                    )
                    await _write_log(
                        db,
                        body,
                        "Cascade",
                        final_model_config,
                        request_context,
                        ms,
                        error=str(cascade_error),
                    )

                if resp and rsl_decision is not None and rsl_decision.recorded:
                    resp.rsl = rsl_decision.response_metadata(
                        executed_model=cascade_meta.final_model
                    )
                if rsl_request_id:
                    await record_outcome_safely(
                        rsl_store,
                        _rsl_outcome(
                            resp,
                            final_model_config,
                            "success" if resp else "error",
                            ms,
                            error=None if resp else RuntimeError("cascade_no_response"),
                            executed_model=cascade_meta.final_model,
                            override_actual_cost=(
                                len(cascade_meta.models_attempted or []) > 1
                                or cascade_meta.judge_model is not None
                            ),
                        ),
                    )

                if cascade_error is None:
                    background_tasks.add_task(
                        cascade_engine.log_cascade,
                        original_model, cascade_meta, 0.0, ms, request_context,
                    )
                else:
                    await cascade_engine.log_cascade(
                        original_model,
                        cascade_meta,
                        0.0,
                        ms,
                        request_context,
                        status="error",
                    )

                if resp:
                    resp.model = original_model
                    if intent_category:
                        resp.intent_category = intent_category
                    return resp
            except ModelRouterError as e:
                if rsl_request_id:
                    await record_outcome_safely(
                        rsl_store,
                        _rsl_outcome(
                            None,
                            chain[-1][2],
                            "error",
                            elapsed(),
                            error=e,
                            executed_model=body.model,
                        ),
                    )
                raise
            except Exception as e:
                ms = elapsed()
                if rsl_request_id:
                    await record_outcome_safely(
                        rsl_store,
                        _rsl_outcome(
                            None,
                            chain[-1][2],
                            "error",
                            ms,
                            error=e,
                            executed_model=body.model,
                        ),
                    )
                background_tasks.add_task(
                    _write_log, db, body, "Cascade",
                    chain[-1][2], request_context, ms, error=str(e),
                )
                raise
            if cascade_error is not None:
                raise cascade_error

    if body.stream:
        try:
            provider, model_config = await router_engine.route(body.model)
        except Exception as route_error:
            attempted_model = body.model
            can_fallback = (
                isinstance(route_error, (ProviderError, AllKeysFailedError))
                or not isinstance(route_error, ModelRouterError)
            )
            if not can_fallback or not _try_rsl_fallback(
                route_error,
                attempted_model,
                "Router",
                router_engine.model_map.get(attempted_model, (None, None))[1],
            ):
                if rsl_request_id:
                    await record_outcome_safely(
                        rsl_store,
                        _rsl_outcome(
                            None,
                            router_engine.model_map.get(attempted_model, (None, None))[1],
                            "error",
                            elapsed(),
                            error=route_error,
                            executed_model=attempted_model,
                        ),
                    )
                raise
            try:
                provider, model_config = await router_engine.route(body.model)
            except Exception as fallback_error:
                if rsl_request_id:
                    await record_outcome_safely(
                        rsl_store,
                        _rsl_outcome(
                            None,
                            router_engine.model_map.get(body.model, (None, None))[1],
                            "error",
                            elapsed(),
                            error=fallback_error,
                            executed_model=body.model,
                        ),
                    )
                raise
        return StreamingResponse(
            _stream(
                body,
                provider,
                model_config,
                db,
                request_context,
                elapsed,
                original_model,
                intent_category,
                router_engine,
                rsl_store,
                rsl_request_id,
                rsl_decision,
                rsl_fallback_model,
            ),
            media_type="text/event-stream",
        )

    retry_history: list[str] = []
    retry_limit = _model_unavailable_retry_limit()
    can_retry_model = (
        recommendation_applied
        and original_model in DEFAULT_MODEL_ALIASES
        and not (cascade_config and cascade_config.get("enabled") is not False)
        and recommender is not None
    )

    while True:
        provider, model_config = await router_engine.route(body.model)
        provider_label = getattr(provider, "provider_name", provider.__class__.__name__)
        attempted_model = body.model
        try:
            resp = await provider.chat_completions(body, model_config)
            ms = elapsed()
            usage = resp.usage
            if body.model != original_model:
                resp.actual_model = body.model
                resp.model = original_model
            if retry_history:
                resp.routing_retry = RoutingRetryMetadata(
                    retry_count=len(retry_history),
                    attempted_models=[*retry_history, body.model],
                    reason="model_temporarily_unavailable",
                )
            if intent_category:
                resp.intent_category = intent_category
            background_tasks.add_task(
                _write_log, db, body, provider_label,
                model_config, request_context, ms, usage,
            )
            if rsl_decision is not None and rsl_decision.recorded:
                resp.rsl = rsl_decision.response_metadata(executed_model=body.model)
            if rsl_request_id:
                background_tasks.add_task(
                    record_outcome_safely,
                    rsl_store,
                    _rsl_outcome(resp, model_config, "success", ms),
                )
            return resp
        except ProviderError as e:
            quarantined = router_engine.mark_model_error(
                attempted_model,
                model_config.name,
                e,
            )
            if _try_rsl_fallback(e, attempted_model, provider_label, model_config):
                continue
            if not quarantined or not can_retry_model or len(retry_history) >= retry_limit:
                await _write_log(
                    db,
                    body,
                    provider_label,
                    model_config,
                    request_context,
                    elapsed(),
                    error=str(e),
                    model_override=attempted_model,
                )
                if rsl_request_id:
                    await record_outcome_safely(
                        rsl_store,
                        _rsl_outcome(None, model_config, "error", elapsed(), error=e),
                    )
                raise

            retry_history.append(attempted_model)
            next_recommendation = recommender.recommend(
                messages=body.messages,
                router_engine=router_engine,
                policy=routing_policy,
                allowed_models=allowed_models,
            )
            next_model = next_recommendation.selected_model if next_recommendation.applied else None
            if not next_model or next_model == attempted_model:
                if rsl_request_id:
                    await record_outcome_safely(
                        rsl_store,
                        _rsl_outcome(None, model_config, "error", elapsed(), error=e),
                    )
                await _write_log(
                    db,
                    body,
                    provider_label,
                    model_config,
                    request_context,
                    elapsed(),
                    error=str(e),
                    model_override=attempted_model,
                )
                raise
            background_tasks.add_task(
                _write_log,
                db,
                body,
                provider_label,
                model_config,
                request_context,
                elapsed(),
                error=str(e),
                model_override=attempted_model,
            )
            body.model = next_model
        except ModelRouterError as e:
            if isinstance(e, AllKeysFailedError) and _try_rsl_fallback(
                e, attempted_model, provider_label, model_config
            ):
                continue
            if rsl_request_id:
                await record_outcome_safely(
                    rsl_store,
                    _rsl_outcome(None, model_config, "error", elapsed(), error=e),
                )
            raise
        except Exception as e:
            if _try_rsl_fallback(e, attempted_model, provider_label, model_config):
                continue
            ms = elapsed()
            if rsl_request_id:
                await record_outcome_safely(
                    rsl_store,
                    _rsl_outcome(None, model_config, "error", ms, error=e),
                )
            background_tasks.add_task(
                _write_log, db, body, provider_label,
                model_config, request_context, ms, error=str(e),
            )
            raise


async def _stream(
    body,
    provider,
    model_config,
    db,
    request_context,
    elapsed,
    original_model=None,
    intent_category=None,
    router_engine=None,
    rsl_store=None,
    rsl_request_id=None,
    rsl_decision=None,
    rsl_fallback_model=None,
):
    usage: Usage | None = None
    error = None
    error_exception = None
    finish_reason = None
    has_content = False
    emitted = False
    completed = False
    pending_chunks = []
    active_provider = provider
    active_model_config = model_config
    baseline_fallback_model = rsl_fallback_model
    fallback_reason = (
        rsl_decision.fallback_reason
        if rsl_decision is not None
        else None
    )

    def _rsl_metadata():
        if rsl_decision is None or not rsl_decision.recorded:
            return None
        return rsl_decision.response_metadata(executed_model=body.model)

    try:
        while True:
            try:
                async for chunk in active_provider.chat_completions_stream(
                    body,
                    active_model_config,
                ):
                    if chunk.usage:
                        usage = chunk.usage
                    if chunk.choices:
                        finish_reason = chunk.choices[0].finish_reason or finish_reason
                        has_content = has_content or bool(chunk.choices[0].delta.content)
                    if body.model != original_model and original_model is not None:
                        chunk.actual_model = body.model
                        chunk.model = original_model
                    if intent_category:
                        chunk.intent_category = intent_category

                    if (
                        not emitted
                        and baseline_fallback_model is not None
                        and not (chunk.choices and chunk.choices[0].delta.content)
                    ):
                        pending_chunks.append(chunk)
                        continue

                    if not emitted:
                        metadata = _rsl_metadata()
                        if metadata and pending_chunks:
                            pending_chunks[0].rsl = metadata
                            metadata = None
                        for pending in pending_chunks:
                            yield f"data: {pending.model_dump_json(exclude_none=True)}\n\n"
                        pending_chunks.clear()
                        if metadata:
                            chunk.rsl = metadata
                        emitted = True
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                if pending_chunks:
                    metadata = _rsl_metadata()
                    if metadata:
                        pending_chunks[0].rsl = metadata
                    for pending in pending_chunks:
                        yield f"data: {pending.model_dump_json(exclude_none=True)}\n\n"
                    pending_chunks.clear()
                completed = True
                yield "data: [DONE]\n\n"
                break
            except Exception as exc:
                if isinstance(exc, ProviderError) and router_engine is not None:
                    router_engine.mark_model_error(body.model, active_model_config.name, exc)
                can_fallback = (
                    baseline_fallback_model is not None
                    and not emitted
                    and (
                        isinstance(exc, (ProviderError, AllKeysFailedError))
                        or not isinstance(exc, ModelRouterError)
                    )
                )
                if not can_fallback:
                    error = str(exc)
                    error_exception = exc
                    metadata = _rsl_metadata()
                    error_payload = {"error": error}
                    if metadata:
                        error_payload["rsl"] = metadata
                    yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                failed_model = body.model
                fallback_reason = type(exc).__name__
                rsl_decision.fallback_reason = fallback_reason
                rsl_decision.details["fallback_from_model"] = failed_model
                await _write_log(
                    db,
                    body,
                    getattr(active_provider, "provider_name", active_provider.__class__.__name__),
                    active_model_config,
                    request_context,
                    elapsed(),
                    error=str(exc),
                    model_override=failed_model,
                )
                pending_chunks.clear()
                baseline_model = baseline_fallback_model
                baseline_fallback_model = None
                body.model = baseline_model
                try:
                    active_provider, active_model_config = await router_engine.route(body.model)
                except Exception as fallback_error:
                    error = str(fallback_error)
                    error_exception = fallback_error
                    metadata = _rsl_metadata()
                    error_payload = {"error": error}
                    if metadata:
                        error_payload["rsl"] = metadata
                    yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                usage = None
                finish_reason = None
                has_content = False
                error_exception = None
    finally:
        if not completed and error is None:
            error = "stream_incomplete"
            error_exception = RuntimeError(error)
        ms = elapsed()
        await _write_log(
            db,
            body,
            getattr(active_provider, "provider_name", active_provider.__class__.__name__),
            active_model_config,
            request_context,
            ms,
            usage,
            error=error,
        )
        if rsl_request_id:
            response_for_outcome = SimpleNamespace(
                usage=usage,
                actual_model=body.model,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=" " if has_content else ""),
                        finish_reason=finish_reason,
                    )
                ],
            )
            outcome = build_outcome(
                request_id=rsl_request_id,
                response=response_for_outcome,
                body=body,
                model_config=active_model_config,
                status="error" if error else "success",
                latency_ms=ms,
                error=error_exception,
                fallback_reason=fallback_reason,
            )
            await record_outcome_safely(rsl_store, outcome)


async def _stream_cascade(
    body, chain, validator, local_judge, webhook_validator, max_escalations,
    db, cascade_engine, request_context, elapsed, original_model, intent_category,
):
    models_attempted = []
    validation_reasons = []
    last_completed_chunks = None
    last_completed_usage = None
    last_provider_name = None
    last_model_config = None
    last_judge_result = None

    for idx, (routing_key, provider, model_config) in enumerate(chain):
        if idx >= max_escalations + 1:
            break

        models_attempted.append(routing_key)
        last_provider_name = getattr(provider, "provider_name", provider.__class__.__name__)
        last_model_config = model_config

        buffered_chunks = []
        usage = None
        error = None

        try:
            async for chunk in provider.chat_completions_stream(body, model_config):
                buffered_chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage
        except Exception as e:
            error = str(e)
            validation_reasons.append(f"{routing_key}:provider_error:{str(e)[:50]}")
            continue

        if error:
            continue
        last_completed_chunks = buffered_chunks
        last_completed_usage = usage

        content = ""
        finish_reason = None
        for chunk in buffered_chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

        temp_resp = ChatResponse(
            id=buffered_chunks[0].id if buffered_chunks else str(uuid.uuid4()),
            created=buffered_chunks[0].created if buffered_chunks else int(time.time()),
            model=routing_key,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )
        temp_req = ChatRequest(
            model=body.model,
            messages=body.messages,
        )
        is_valid, reason, judge_result = await cascade_engine.evaluate_response(
            temp_resp,
            temp_req,
            validator,
            local_judge,
            webhook_validator,
        )
        last_judge_result = judge_result

        validation_reasons.append(f"{routing_key}:{reason}")

        if is_valid:
            cascade_meta = CascadeMetadata(
                enabled=True,
                level=idx,
                escalated_from=original_model if idx > 0 else None,
                final_model=routing_key,
                validation_passed=True,
                escalation_reason=validation_reasons[0] if idx > 0 else None,
                models_attempted=models_attempted,
                judge_model=local_judge.judge_model if local_judge else None,
                judge_score=judge_result.score if judge_result else None,
                judge_reason=judge_result.reason if judge_result else None,
            )

            for chunk in buffered_chunks:
                if body.model != original_model and original_model is not None:
                    chunk.actual_model = body.model
                    chunk.model = original_model
                if intent_category:
                    chunk.intent_category = intent_category
                chunk.cascade = cascade_meta
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

            ms = elapsed()
            if usage:
                await _write_log(
                    db, body, last_provider_name,
                    last_model_config, request_context, ms, usage,
                )
            await cascade_engine.log_cascade(
                original_model, cascade_meta, 0.0, ms, request_context,
            )
            return

        if idx < len(chain) - 1:
            logger = __import__("logging").getLogger("model_router.cascade")
            logger.info(
                "Cascade streaming escalation: %s -> %s (reason: %s)",
                routing_key, chain[idx + 1][0], reason,
            )

    if last_completed_chunks:
        cascade_meta = CascadeMetadata(
            enabled=True,
            level=len(models_attempted) - 1,
            escalated_from=original_model if len(models_attempted) > 1 else None,
            final_model=models_attempted[-1] if models_attempted else original_model,
            validation_passed=False,
            escalation_reason="; ".join(validation_reasons),
            models_attempted=models_attempted,
            judge_model=local_judge.judge_model if local_judge else None,
            judge_score=last_judge_result.score if last_judge_result else None,
            judge_reason=last_judge_result.reason if last_judge_result else None,
        )
        for chunk in last_completed_chunks:
            if body.model != original_model and original_model is not None:
                chunk.actual_model = body.model
                chunk.model = original_model
            if intent_category:
                chunk.intent_category = intent_category
            chunk.cascade = cascade_meta
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
    else:
        error_msg = {
            "error": "All cascade models failed",
            "details": "; ".join(validation_reasons),
        }
        yield f"data: {json.dumps(error_msg)}\n\n"
        yield "data: [DONE]\n\n"

    ms = elapsed()
    if last_completed_chunks and last_completed_usage:
        await _write_log(
            db, body, last_provider_name,
            last_model_config, request_context, ms, last_completed_usage,
        )
    elif last_provider_name and last_model_config:
        await _write_log(
            db,
            body,
            last_provider_name,
            last_model_config,
            request_context,
            ms,
            error="All cascade models failed",
        )
    cascade_meta = CascadeMetadata(
        enabled=True,
        level=len(models_attempted) - 1,
        escalated_from=original_model if len(models_attempted) > 1 else None,
        final_model=models_attempted[-1] if models_attempted else original_model,
        validation_passed=False,
        escalation_reason="; ".join(validation_reasons),
        models_attempted=models_attempted,
        judge_model=local_judge.judge_model if local_judge else None,
        judge_score=last_judge_result.score if last_judge_result else None,
        judge_reason=last_judge_result.reason if last_judge_result else None,
    )
    await cascade_engine.log_cascade(
        original_model, cascade_meta, 0.0, ms, request_context, status="error",
    )
