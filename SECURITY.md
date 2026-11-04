# Security policy

## Supported versions

Security fixes land on the current major line (`3.x`), in a patch release (`3.0.x`).
Older majors (`0.x` through `2.x`) are unmaintained and will not receive a fix;
upgrade to `3.x` before reporting.

| Version | Supported |
| --- | --- |
| `3.x` | yes |
| `0.x` – `2.x` | no |

## Reporting a vulnerability

**Do not open a public issue, pull request or discussion for a suspected
vulnerability.** A public report is already a disclosure — including a "just a quick
question, is this a security bug?" issue that quotes the payload.

Report it privately through GitHub Security Advisories:

<https://github.com/ravenkernel/chassis-sim/security/advisories/new>

That is a private channel between you and the maintainers, and it gives us a place to
discuss the issue, agree on a fix and publish a credit — without exposing users in the
meantime. Please include:

- the affected version(s) (`python -c "import chassis_sim; print(chassis_sim.__version__)"`),
  or the tag you checked out;
- your Python version and operating system;
- a minimal reproduction: the smallest script that shows the problem, plus any YAML
  config or scenario file it needs;
- the impact you believe it has, and whether you plan to publish anything about it.

If the advisory form is unavailable to you, open a public issue that says only
"security issue, please provide a private channel" and no technical detail; a
maintainer will open an advisory and move the conversation there.

## What to expect

| Stage | Target |
| --- | --- |
| acknowledgement of the report | within 7 days |
| initial assessment (severity, affected versions, whether a fix is planned) | within 14 days |
| fix and advisory publication | in the next patch release after a fix is agreed |

This is a small volunteer project, so those are targets rather than a contractual
response time. We will tell you if a report is out of scope, and we will credit you in
the advisory unless you ask us not to.

## Scope

chassis-sim is a local simulator: it opens no network connections and evaluates no
untrusted code from remote input. In scope is anything reachable from the public
`chassis_sim` API or the `chassis-sim` command-line entry point that does more than the
documentation says — a crash on ordinary input, a path read or written outside the
directory the caller pointed at, or code execution driven by a config or scenario
file. Out of scope: a simulation that is physically implausible, a missing feature, and
vulnerabilities in `numpy`, `gymnasium` or `matplotlib` themselves (report those
upstream — tell us as well if the fix changes how we pin them).
