# Hosted AuthKit resource server (V2 TEST deployment validated)

The hosted server validates WorkOS AuthKit access tokens directly. WorkOS owns
authorization, login/consent, PKCE, CIMD/DCR and refresh. The MCP server does not
issue reference tokens or host authorization, registration, callback or token
exchange endpoints. Clients must reconnect when migrating from the old proxy.

## Required server configuration

- `ALPACA_MCP_AUTHKIT_ISSUER`: exact HTTPS AuthKit origin, without a trailing slash,
  path, query or fragment. This must equal the issuer in WorkOS metadata and JWTs.
- `ALPACA_MCP_PUBLIC_BASE_URL`: canonical public HTTPS origin, without a trailing
  slash. No real deployment hostname is supplied by the application.
- `ALPACA_MCP_OIDC_AUDIENCE`: exact public MCP endpoint URL. For the V2 TEST HTTP
  endpoint this is the configured public base URL plus `/mcp`.
- `ALPACA_MCP_OAUTH_SCOPES`: nonempty, space-separated required OAuth scopes.
  There is no permissive default and no automatic fallback to `openid`.
- `DATABASE_URL`: still required for the durable Safe Trading risk ledger.
- `ALPACA_SAFE_OWNERSHIP_SECRET`: still required and validated as before.
- `ALPACA_PAPER_TRADE`: must be the exact string `true` for hosted startup or
  construction of a trading client. Missing, empty, whitespace-padded,
  case-variant, `false`, `1`, `yes`, and unknown values fail closed. A non-hosted
  market-data-only configuration does not construct a trading client; this is
  not a write-permission exception.

The factory retains its existing `build_managed_oidc_provider` name to minimize
call-site churn, but returns `HardenedAuthKitProvider`, not OIDCProxy.

Upstream OAuth client ID/secret, discovery config URL and the MCP proxy JWT
signing key are no longer used by this provider. It constructs no OAuth storage.
Existing OAuth tables are NOT dropped or migrated. Cleanup requires separate
approval. The AuthKit migration does not change risk tables or HMAC ownership.
The current write and reconciliation safeguards are described below.

## Security contract

Only the default trusted FastMCP JWT verifier is used. Its signature, issuer,
audience, scope and strict expiration checks precede the project's finite
NumericDate validation and SDK subject binding. `exp` is mandatory; `nbf` and
`iat` are optional and cannot be more than 60 seconds in the future. Booleans,
strings, null, containers and non-finite numbers are rejected. The additional
gate never revives a token rejected upstream as expired. No maximum lifetime
policy is invented. JWKS/verification failures fail closed.

The issuer is not normalized from token data. Protected-resource metadata emits
the exact configured issuer without the trailing slash that generic URL models
would add. The principal remains `oauth-v2-` plus SHA256 of verified issuer,
NUL, and verified subject. Token refresh must preserve issuer/sub to preserve
the principal; sessions additionally retain the SDK's client/issuer/subject binding.

For `/mcp`, metadata is at `/.well-known/oauth-protected-resource/mcp` and the
401 challenge points there. Its resource and the JWT audience must match the
actual public MCP endpoint; mount mismatch fails during app construction.
`/.well-known/oauth-authorization-server` only forwards WorkOS discovery metadata.
There is no server-side refresh or token storage. MCP transport settings are unchanged;
this does not claim MCP 2026-07-28 support.

## Enforced Safe Trading boundaries

The trading client is constructed with the fixed Alpaca Paper endpoint
`https://paper-api.alpaca.markets`. Safe handlers independently check the actual
client base URL (the endpoint with or without its root slash) and require
redirect following to be disabled. The check is repeated before POST/DELETE;
a Paper boolean alone is not destination proof. There is no configuration
switch to select a Live Safe write client.

Normal hosted, HTTP and stdio/non-hosted registration cannot restore legacy/raw
writes, including with `ALPACA_SAFE_MODE=false` or different toolset combinations.
When the trading toolset is selected, the write surface is exactly:

- `safe_place_stock_order`
- `safe_place_crypto_order`
- `safe_cancel_order`

`safe_close_position` remains absent. Toolsets without trading do not gain Safe
writes. Existing read-only tools retain their normal toolset selection.

The stock tool is equity-only: it rejects OCC option symbols, then requires a
successful read-only `GET /v2/assets/{symbol}` preflight. The returned symbol
must match; `class` or `asset_class` must establish `us_equity`, with no
conflicting class fields; `status` must be `active` and `tradable` must be true.
Missing, malformed, ambiguous, non-equity or failed responses reject the order.
After authorization, Paper transport and input checks, this preflight happens
before reconciliation, RiskStore reservation or order POST. Failed asset
validation therefore creates no reservation.

Cancellation checks the broker order ID, server-derived client_order_id, exact
canonical symbol, durable HMAC ownership, principal and strategy. Symbol
mismatch also rejects uncertain-order binding. DELETE 204 means only that the
cancel request was accepted: the ledger retains `cancel_uncertain` and all
open/reserved risk. Repeated cancellation does not blindly repeat DELETE.
Later startup/on-write reconciliation must positively verify an owned order's
recognized terminal broker state before releasing open exposure. Pending,
unknown, malformed, missing, timeout and other ambiguous results retain risk;
redirects are not terminal confirmation. Confirmed cancellation does not refund
daily submitted notional. No unbounded cancellation wait is introduced.

Safe symbols use bounded ASCII formats (maximum 32 characters); strategy IDs
and idempotency keys use 1–64 allowed ASCII characters. Numeric inputs must be
ASCII decimal strings of at most 64 characters, with at most 12 coefficient
digits and decimal exponent/adjusted magnitude between -12 and 12. Invalid
types, non-finite numbers and pathological expansions are rejected. Accepted
values use exact Decimal arithmetic and canonical fixed-point request text,
without silently rounding intent; the same request text feeds the fingerprint.
The $100 order ceiling, $10 crypto minimum and cumulative ceilings are unchanged.

Regression tests never infer cleanup authorization from Alpaca credentials.
Automatic account cancellation/position cleanup has been removed entirely;
any broker cleanup requires a separately authorized operator action.

## V2 TEST validation and release boundaries

V2 TEST deployment and end-to-end Paper smoke validation are complete. Validated
in V2 TEST: real WorkOS AuthKit integration, Claude hosted custom connector
interoperability, CIMD/PKCE login, the exact MCP resource/audience path, managed
PostgreSQL RiskStore, and the signed JWT Template claim `alpaca_role="paper-trader"`.

The authorization gate and deterministic $10 crypto minimum rejection were
validated. A real Paper $20 BTC/USD LIMIT order used a server-derived `safe-v2-`
client_order_id and durable ownership; exact owned cancellation completed with
final status `canceled`, `filled_qty=0`, and no BTC/USD position. Temporary
authentication diagnostics were removed. Post-cleanup read-only BTC/USD and
AAPL calls also passed.

Those external smoke results predate the latest local security remediation;
they do not claim that the current uncommitted patch has been deployed or
revalidated against a real broker.

These results validate V2 TEST, not Live readiness or production approval.
Live trading / production-live deployment is NOT approved. Final security/release
review is still required before merge. This smoke test does not establish every
token-refresh, revocation, restart, or JWKS failure/rotation scenario. DCR is not
claimed validated by the CIMD result. Authentication remains separate from trading
authorization; required scopes must not be weakened to make login succeed.
Paper-only and exactly three Safe writes remain; close-position remains disabled.

## Paper-write authorization

Authentication alone does not authorize trading. Each of the three Safe V2 write
handlers calls `has_paper_trading_permission()` before validation, reconciliation,
principal lookup, RiskStore access, or broker access. It grants capability only
from the current verified access-token claims dictionary through either path:

- Native `permissions`: a non-empty list whose every item is a non-empty trimmed
  string, containing the exact string `paper-trading`.
- Custom signed `alpaca_role`: the exact string `paper-trader`, trusted only after
  normal WorkOS/FastMCP JWT verification has succeeded.

There is no trimming, lowercasing, normalization, coercion, or alias mapping of
role values. Native `role` and `roles` are not authorization sources. Neither are
`org_id`, scopes, email, `sub`, `sid`, `strategy_id`, tool arguments, client
assertions, request parameters, or unverified JWT payloads.

Missing context or non-dictionary claims deny writes, including local/stdio calls.
Missing or malformed authorization claims do not grant capability: one path must
independently satisfy its exact rule. An invalid native permissions claim does
not veto an independently valid signed custom role, and vice versa. Processing
exceptions fail closed unless an allowed path has already safely returned true.
Read-only tools retain the existing authentication requirements without this gate.
All existing Paper-only, ownership and cumulative-risk controls still apply.

Roles and permissions are a signed JWT snapshot, not a live WorkOS lookup. Removing a role
or permission does not immediately revoke an already-issued token. Clients must
obtain and present updated tokens; no immediate session revocation is promised.
The real Connect flow validated the custom signed role path; native `role` and
`permissions` were absent in that observed token. The native permissions path
remains supported and covered by local regression tests. Role/permission changes
may require token refresh; V2 TEST success does not imply immediate revocation.
