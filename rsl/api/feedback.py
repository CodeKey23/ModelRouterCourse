from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.auth import resolve_request_context

router = APIRouter(tags=["rsl / feedback"])


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    rating: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = Field(default=None, min_length=1, max_length=64)


@router.post("/v1/feedback")
async def submit_feedback(body: FeedbackRequest, request: Request):
    if not getattr(request.app.state, "rsl_recording_enabled", False):
        raise HTTPException(status_code=404, detail="RSL feedback is disabled")

    store = getattr(request.app.state, "rsl_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="RSL store is unavailable")

    label = body.label.strip() if body.label else None
    if body.rating is None and not label:
        raise HTTPException(
            status_code=400,
            detail="At least one of rating or label is required",
        )

    context = await resolve_request_context(request)
    decision = await store.get_decision(body.request_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="RSL request was not found")

    if context.subject_type == "internal":
        if not context.user_id or decision.get("user_id") != context.user_id:
            raise HTTPException(status_code=403, detail="Feedback identity does not match request")
    elif (
        decision.get("subject_type") != context.subject_type
        or decision.get("subject_id") != context.subject_id
    ):
        raise HTTPException(status_code=403, detail="Feedback identity does not match request")

    await store.save_feedback(
        request_id=body.request_id,
        subject_type=context.subject_type,
        subject_id=context.subject_id,
        rating=body.rating,
        label=label,
        source=context.subject_type,
    )
    return {"ok": True, "request_id": body.request_id}
