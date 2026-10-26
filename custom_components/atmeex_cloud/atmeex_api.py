"""Client for the Atmeex cloud API (api.iot.atmeex.com)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generator

import httpx

_LOGGER = logging.getLogger(__name__)

ATMEEX_API_BASE_URL = "https://api.iot.atmeex.com"
USER_AGENT = "okhttp/3.14.9"
COMMON_HEADERS = {"accept": "application/json", "user-agent": USER_AGENT}
HEATER_DISABLE_TEMP = -1000

_NOT_AUTHENTICATED_MSG = (
    "Not authenticated. Call client.signin_with_email() "
    "or client.request_sms_code() + signin_with_phone() first"
)


class AtmeexAuthError(Exception):
    """Raised when authentication fails and cannot be recovered."""


@dataclass
class DeviceSettings:
    """Writable device settings returned by the cloud API."""

    u_pwr_on: bool
    u_fan_speed: int
    u_damp_pos: int
    u_temp_room: int
    u_hum_stg: int = 0
    u_auto: bool = False
    u_night: bool = False
    u_night_start: str = ""
    u_night_stop: str = ""
    u_cool_mode: bool | None = None
    u_time_zone: int | None = None


@dataclass
class DeviceCondition:
    """Last reported device telemetry. Fields vary between API revisions."""

    time: str | None = None
    pwr_on: bool | None = None
    no_water: bool | None = None
    co2_ppm: int | None = None
    temp_in: int | None = None
    temp_room: int | None = None
    fan_speed: int | None = None
    damp_pos: int | None = None
    hum_room: int | None = None
    hum_stg: int | None = None
    firmware_version: str | None = None
    server_time: str | None = None
    device_id: int | None = None
    created_at: str | None = None


@dataclass
class DeviceModel:
    """Device description used by the Home Assistant entities."""

    id: int
    name: str
    model: str
    settings: DeviceSettings
    mac: str = ""
    type: int = 0
    online: bool | None = None
    room_id: int | None = None
    owner_id: int | None = None
    created_at: str | None = None
    fw_ver: str | None = None
    socket_id: str | None = None
    condition: DeviceCondition | None = None


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_int(value: Any) -> int:
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def parse_settings(data: dict[str, Any]) -> DeviceSettings:
    """Parse device settings from either the current or the legacy payload."""
    return DeviceSettings(
        u_pwr_on=_as_bool(data["u_pwr_on"]),
        u_fan_speed=_as_int(data["u_fan_speed"]),
        u_damp_pos=_as_int(data["u_damp_pos"]),
        u_temp_room=_as_int(data["u_temp_room"]),
        u_hum_stg=_as_int(data.get("u_hum_stg", 0)),
        u_auto=_as_bool(data.get("u_auto", False)),
        u_night=_as_bool(data.get("u_night", False)),
        u_night_start=str(data.get("u_night_start") or ""),
        u_night_stop=str(data.get("u_night_stop") or ""),
        u_cool_mode=_optional_bool(data.get("u_cool_mode")),
        u_time_zone=_optional_int(data.get("u_time_zone")),
    )


def parse_condition(data: dict[str, Any] | None) -> DeviceCondition | None:
    """Parse telemetry. Missing legacy fields are left empty."""
    if not isinstance(data, dict):
        return None

    return DeviceCondition(
        time=data.get("time"),
        pwr_on=_optional_bool(data.get("pwr_on")),
        no_water=_optional_bool(data.get("no_water")),
        co2_ppm=_optional_int(data.get("co2_ppm")),
        temp_in=_optional_int(data.get("temp_in")),
        temp_room=_optional_int(data.get("temp_room")),
        fan_speed=_optional_int(data.get("fan_speed")),
        damp_pos=_optional_int(data.get("damp_pos")),
        hum_room=_optional_int(data.get("hum_room")),
        hum_stg=_optional_int(data.get("hum_stg")),
        firmware_version=data.get("firmware_version"),
        server_time=data.get("server_time"),
        device_id=_optional_int(data.get("device_id")),
        created_at=data.get("created_at"),
    )


def parse_device(data: dict[str, Any]) -> DeviceModel:
    """Parse one device object from GET /devices or PUT /devices/{id}/params."""
    if not isinstance(data, dict):
        raise ValueError(f"Device payload must be an object, got {type(data).__name__}")

    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"Device {data.get('id')} has no settings object")

    condition = parse_condition(data.get("condition"))
    firmware = data.get("fw_ver")
    if not firmware and condition is not None:
        firmware = condition.firmware_version

    return DeviceModel(
        id=_as_int(data["id"]),
        name=str(data.get("name") or f"Atmeex {data['id']}"),
        model=str(data.get("model") or ""),
        settings=parse_settings(settings),
        mac=str(data.get("mac") or ""),
        type=_as_int(data.get("type") or 0),
        online=_optional_bool(data.get("online")),
        room_id=_optional_int(data.get("room_id")),
        owner_id=_optional_int(data.get("owner_id")),
        created_at=data.get("created_at"),
        fw_ver=str(firmware) if firmware else None,
        socket_id=data.get("socket_id"),
        condition=condition,
    )


class AtmeexAuth(httpx.Auth):
    """Bearer auth with refresh-token retry."""

    requires_response_body = True

    def __init__(self, access_token: str = "", refresh_token: str = "") -> None:
        self._access_token_value = access_token
        self._refresh_token_value = refresh_token

    @property
    def access_token(self) -> str:
        return self._access_token_value

    @access_token.setter
    def access_token(self, value: str) -> None:
        self._access_token_value = value

    @property
    def refresh_token(self) -> str:
        return self._refresh_token_value

    @refresh_token.setter
    def refresh_token(self, value: str) -> None:
        self._refresh_token_value = value

    @staticmethod
    def request_phone_code(phone: str) -> httpx.Request:
        payload = {
            "grant_type": "phone_code",
            "phone": phone,
        }
        return httpx.Request(
            "POST",
            ATMEEX_API_BASE_URL + "/auth/signup",
            json=payload,
            headers=COMMON_HEADERS,
        )

    @staticmethod
    def signin_with_phone(phone: str, phone_code: str) -> httpx.Request:
        payload = {
            "grant_type": "phone_code",
            "phone": phone,
            "phone_code": phone_code,
        }
        return httpx.Request(
            "POST",
            ATMEEX_API_BASE_URL + "/auth/signin",
            json=payload,
            headers=COMMON_HEADERS,
        )

    @staticmethod
    def signin_with_email(email: str, password: str) -> httpx.Request:
        payload = {
            "email": email,
            "password": password,
            "grant_type": "basic",
        }
        return httpx.Request(
            "POST",
            ATMEEX_API_BASE_URL + "/auth/signin",
            json=payload,
            headers=COMMON_HEADERS,
        )

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        if "/auth/" in str(request.url):
            yield request
            return

        if self._access_token_value == "":
            raise AtmeexAuthError(_NOT_AUTHENTICATED_MSG)

        request.headers["authorization"] = f"Bearer {self._access_token_value}"
        response = yield request

        if response.status_code == 401:
            yield from self._refresh_token()
            request.headers["authorization"] = f"Bearer {self._access_token_value}"
            yield request

    def _refresh_token(self) -> Generator[httpx.Request, httpx.Response, None]:
        if self._refresh_token_value == "":
            raise AtmeexAuthError(_NOT_AUTHENTICATED_MSG)

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token_value,
        }
        response = yield httpx.Request(
            "POST",
            ATMEEX_API_BASE_URL + "/auth/signin",
            json=payload,
            headers=COMMON_HEADERS,
        )
        if response.status_code == 401:
            raise AtmeexAuthError("Session expired. Please sign in again.")
        self._handle_auth_response(response)

    def _handle_auth_response(self, response: httpx.Response) -> None:
        response.raise_for_status()
        data = response.json()
        self._refresh_token_value = data["refresh_token"]
        self._access_token_value = data["access_token"]


class Device:
    """One Atmeex device and the commands that change its settings."""

    def __init__(self, http_client: httpx.AsyncClient, data: dict[str, Any]) -> None:
        self._http_client = http_client
        self.model = parse_device(data)

    async def _set_params(self, **fields: Any) -> None:
        payload = {key: value for key, value in fields.items() if value is not None}
        if not payload:
            raise ValueError("Can't update device settings when every field is empty")

        response = await self._http_client.put(
            f"/devices/{self.model.id}/params", json=payload
        )
        response.raise_for_status()
        self.model = parse_device(response.json())

    async def set_heat_temp(self, temp: int) -> None:
        temp = int(temp)
        if (temp < 100 or temp > 300) and temp != HEATER_DISABLE_TEMP:
            raise ValueError(
                f"set_heat_temp temp not between 100 and 300 and not -1000: {temp}"
            )
        if temp % 10 not in (0, 5):
            raise ValueError(f"set_heat_temp temp ends not with 0 or 5: {temp}")

        await self._set_params(u_temp_room=temp)

    async def set_fan_speed(self, fan_speed: int) -> None:
        if fan_speed < 0 or fan_speed > 6:
            raise ValueError(f"set_fan_speed fan_speed not between 0 and 6: {fan_speed}")

        await self._set_params(u_fan_speed=fan_speed)

    async def set_power_and_damp(self, power: bool, damp_pos: int) -> None:
        await self._set_params(u_damp_pos=damp_pos, u_pwr_on=power)

    async def set_power_only(self, power: bool) -> None:
        """Set power without changing damper position."""
        await self._set_params(u_pwr_on=power)

    async def set_damp_pos(self, damp_pos: int) -> None:
        """Set damper position: 0=open, 1=mixed, 2=closed."""
        if damp_pos not in (0, 1, 2):
            raise ValueError(f"set_damp_pos damp_pos must be 0, 1 or 2: {damp_pos}")

        await self._set_params(u_damp_pos=damp_pos)


class AtmeexClient:
    """HTTP client for api.iot.atmeex.com."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self.auth = AtmeexAuth()
        if http_client is not None:
            self.http_client = http_client
            self.http_client.auth = self.auth
            self.http_client.headers.update(COMMON_HEADERS)
            self.http_client.base_url = ATMEEX_API_BASE_URL
        else:
            self.http_client = httpx.AsyncClient(
                auth=self.auth, headers=COMMON_HEADERS, base_url=ATMEEX_API_BASE_URL
            )

    async def request_sms_code(self, phone: str) -> None:
        """Send SMS code to phone number."""
        request = AtmeexAuth.request_phone_code(phone)
        response = await self.http_client.send(request)
        response.raise_for_status()

    async def signin_with_phone(self, phone: str, code: str) -> None:
        """Sign in with phone and SMS code."""
        request = AtmeexAuth.signin_with_phone(phone, code)
        response = await self.http_client.send(request)
        response.raise_for_status()
        data = response.json()
        self.auth.access_token = data["access_token"]
        self.auth.refresh_token = data["refresh_token"]

    async def signin_with_email(self, email: str, password: str) -> None:
        """Sign in with email and password."""
        request = AtmeexAuth.signin_with_email(email, password)
        response = await self.http_client.send(request)
        response.raise_for_status()
        data = response.json()
        self.auth.access_token = data["access_token"]
        self.auth.refresh_token = data["refresh_token"]

    def restore_tokens(self, access_token: str, refresh_token: str) -> None:
        """Restore session from saved tokens."""
        self.auth.access_token = access_token
        self.auth.refresh_token = refresh_token

    @property
    def access_token(self) -> str | None:
        return self.auth.access_token

    @property
    def refresh_token(self) -> str | None:
        return self.auth.refresh_token

    async def get_devices(self) -> list[Device]:
        """Return devices linked to the account."""
        response = await self.http_client.get("/devices")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(
                "Unexpected /devices payload type: "
                f"{type(payload).__name__}"
            )

        devices: list[Device] = []
        for item in payload:
            device_id = item.get("id") if isinstance(item, dict) else None
            try:
                devices.append(Device(self.http_client, item))
            except Exception:
                _LOGGER.exception("Failed to parse Atmeex device %s", device_id)

        if payload and not devices:
            raise ValueError("API returned devices but none could be parsed")

        return devices
