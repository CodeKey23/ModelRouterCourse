import json

from app.core.exceptions import ProviderError


def decode_sse_payload(raw: str, provider_name: str, event: str = "") -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(provider_name, 502, f"invalid stream JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ProviderError(provider_name, 502, "stream event must contain a JSON object")

    error = payload.get("error")
    if event == "error" or error:
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or str(error)
            status = error.get("status_code") or error.get("status")
        else:
            message = error or payload.get("message") or "upstream stream error"
            status = payload.get("status_code") or payload.get("status")
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            status_code = 502
        if status_code < 400 or status_code > 599:
            status_code = 502
        raise ProviderError(provider_name, status_code, str(message)[:300])

    return payload


def parse_stream_chunk(parser, payload: dict, model: str, provider_name: str):
    try:
        return parser(payload, model)
    except ProviderError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError(
            provider_name,
            502,
            f"invalid stream response: {type(exc).__name__}",
        ) from exc


def normalize_stream_error(error: Exception, provider_name: str) -> ProviderError:
    return ProviderError(provider_name, 0, f"stream transport error: {error}")
