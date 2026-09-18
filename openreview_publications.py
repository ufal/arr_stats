"""
OpenReview Publication Fetcher (2025–2026)
==========================================
Fetches all your submissions from OpenReview, filters to 2025–2026,
groups by conference/venue, and extracts acceptance status.

Special handling for ACL Rolling Review (ARR) papers that commit
to adjacent conferences (ACL, EMNLP, NAACL, EACL, COLING, etc.).

Usage:
    pip install openreview-py
    python openreview_publications.py --username YOUR_EMAIL --password YOUR_PASSWORD

    Or set environment variables:
        OPENREVIEW_USERNAME=your@email.com
        OPENREVIEW_PASSWORD=yourpassword
"""

import os
import re
import sys
import argparse
from collections import defaultdict
from datetime import datetime, timezone

try:
    import openreview
except ImportError:
    sys.exit(
        "openreview-py is not installed. Run:\n"
        "    pip install openreview-py\n"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Venues that are part of the ACL Rolling Review ecosystem.
# ARR itself, plus the *ACL conferences that accept ARR commitments.
ARR_VENUES = {
    "aclweb.org/ACL",
    "aclweb.org/EMNLP",
    "aclweb.org/NAACL",
    "aclweb.org/EACL",
    "aclweb.org/COLING",
    "aclweb.org/Findings",
    "aclweb.org/ACL/ARR",
    "aclweb.org/ARR",
}

# Matches the ARR rolling review itself: aclweb.org/ACL/ARR/YYYY/Month/...
ARR_SUBMISSION_PATTERN = re.compile(r"aclweb\.org/ACL/ARR/\d{4}/\w+")

# Matches papers committed to a venue via ARR: ..._ARR_Commitment/...
ARR_COMMITMENT_PATTERN = re.compile(r"_ARR_Commitment")

# Explicit allowlist of *ACL conference venue prefixes (not workshops, not SRW)
ACL_MAIN_CONFERENCE_PREFIXES = (
    "aclweb.org/ACL/",
    "aclweb.org/EMNLP/",
    "aclweb.org/NAACL/",
    "aclweb.org/EACL/",
    "aclweb.org/COLING/",
    "aclweb.org/AACL",        # covers AACL and AACL-IJCNLP
    "aclweb.org/IJCNLP/",
    "aclweb.org/CL/",
    "aclweb.org/TACL/",
)

# Decision field names OpenReview uses (varies by venue)
DECISION_FIELDS = [
    "decision",
    "recommendation",
    "Decision",
    "Recommendation",
    "acceptance",
    "paper_decision",
    "desk_reject_comments",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts_to_year(timestamp_ms: int | None) -> int | None:
    """Convert a millisecond timestamp to a year."""
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).year


def ts_to_date(timestamp_ms: int | None) -> str:
    """Convert a millisecond timestamp to a human-readable date string."""
    if timestamp_ms is None:
        return "Unknown"
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def is_acl_main_conference(venue_id: str) -> bool:
    """True for *ACL main conferences (not workshops, SRW, or unrelated venues)."""
    if not any(venue_id.startswith(prefix) for prefix in ACL_MAIN_CONFERENCE_PREFIXES):
        return False
    lowered = venue_id.lower()
    return not any(x in lowered for x in ("workshop", "srw", "tutorial", "industry_track", "demo"))


# Bare venue IDs that lack the aclweb.org prefix but are still *ACL main conferences
ACL_BARE_PREFIXES = (
    "EMNLP/",
    "NAACL/",
    "EACL/",
    "COLING/",
    "AACL/",
)


def is_arr_venue(venue_id: str) -> bool:
    """True for ARR submissions, and *ACL main conference papers (direct or via commitment)."""
    if ARR_SUBMISSION_PATTERN.search(venue_id):
        return True
    if ARR_COMMITMENT_PATTERN.search(venue_id) and is_acl_main_conference(venue_id):
        return True
    if is_acl_main_conference(venue_id):
        return True
    # Some venues appear without the aclweb.org prefix
    if any(venue_id.startswith(p) for p in ACL_BARE_PREFIXES):
        lowered = venue_id.lower()
        if not any(x in lowered for x in ("workshop", "srw", "tutorial", "industry_track", "demo")):
            return True
    return False


def extract_decision(note: openreview.Note) -> str:
    """Try to find an acceptance decision in a note's content."""
    content = note.content or {}

    for field in DECISION_FIELDS:
        if field in content:
            val = content[field]
            if isinstance(val, dict):
                val = val.get("value", "")
            if val:
                return str(val).strip()

    # Sometimes the venue field itself carries accept/reject info
    venue = content.get("venue", {})
    if isinstance(venue, dict):
        venue = venue.get("value", "")
    if venue:
        lower = venue.lower()
        if "reject" in lower:
            return "Reject"
        if "accept" in lower or "findings" in lower:
            return "Accept"

    return "Unknown"


def get_paper_title(note: openreview.Note) -> str:
    content = note.content or {}
    title = content.get("title", {})
    if isinstance(title, dict):
        return title.get("value", "Untitled")
    return str(title) if title else "Untitled"



def get_authors(note: openreview.Note) -> list[str]:
    content = note.content or {}
    authors = content.get("authors", {})
    if isinstance(authors, dict):
        authors = authors.get("value", [])
    if not authors:
        return []
    result = []
    for a in authors:
        if isinstance(a, dict):
            result.append(a.get("fullname") or a.get("name") or str(a))
        else:
            result.append(str(a))
    return result


def get_metareview_score(note: openreview.Note) -> str | None:
    """Extract 'Overall assessment' from the first meta-review reply, if present."""
    replies = getattr(note, "details", {}) or {}
    for r in replies.get("directReplies", []):
        if isinstance(r, dict):
            invs = r.get("invitations", [r.get("invitation", "")])
            rc = r.get("content", {})
        else:
            invs = getattr(r, "invitations", None) or [getattr(r, "invitation", "") or ""]
            rc = getattr(r, "content", None) or {}
        inv = invs[0] if invs else ""
        if not re.search(r"[Mm]eta.?[Rr]eview", inv):
            continue
        for key in ("overall_assessment", "Overall Assessment", "overall assessment",
                    "recommendation", "Recommendation"):
            if key in rc:
                val = rc[key]
                if isinstance(val, dict):
                    val = val.get("value", "")
                if val:
                    return str(val).strip()
    return None

def get_venue_label(invitation: str, content: dict) -> str:
    """Return a clean venue label from the invitation string."""
    # e.g. "aclweb.org/ACL/2025/Conference/-/Submission"
    # Strip the trailing action part
    venue = invitation.split("/-/")[0] if "/-/" in invitation else invitation
    # Prefer the human-readable venue name from content if available
    venue_name = content.get("venue", {})
    if isinstance(venue_name, dict):
        venue_name = venue_name.get("value", "")
    if venue_name:
        return str(venue_name)
    return venue


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def fetch_submissions(client: openreview.api.OpenReviewClient, profile_id: str):
    """Fetch all submissions where the logged-in user is an author."""
    print(f"Fetching submissions for profile: {profile_id}")
    notes = []

    # v2 API: search by author tilde-id or email
    # `authorids` field stores tilde IDs like ~Firstname_Lastname1
    try:
        # Try fetching via tilde ID (preferred)
        results = client.get_all_notes(
            content={"authorids": profile_id},
            details="original,directReplies",
        )
        notes.extend(results)
        print(f"  Found {len(results)} submissions via tilde ID.")
    except Exception as e:
        print(f"  Warning: tilde-ID search failed ({e}), trying email…")

    # Also try by email in case profile uses both
    if "@" in profile_id:
        try:
            results_email = client.get_all_notes(
                content={"authorids": profile_id},
                details="original,directReplies",
            )
            # Deduplicate by note id
            existing_ids = {n.id for n in notes}
            for r in results_email:
                if r.id not in existing_ids:
                    notes.append(r)
                    existing_ids.add(r.id)
            print(f"  Found {len(results_email)} additional submissions via email.")
        except Exception as e:
            print(f"  Email search also failed: {e}")

    return notes


def get_decision_from_replies(note: openreview.Note, client: openreview.api.OpenReviewClient) -> str:
    """
    Look through direct replies for an accept/reject decision.
    Prefer replies whose invitation looks like a final decision
    (e.g. .../Decision, .../Paper_Decision) over meta-reviews.
    """
    replies = getattr(note, "details", {}) or {}
    direct_replies = replies.get("directReplies", [])

    def reply_invitation(r) -> str:
        if isinstance(r, dict):
            invs = r.get("invitations", [])
            return invs[0] if invs else r.get("invitation", "")
        invs = getattr(r, "invitations", None) or []
        return invs[0] if invs else (getattr(r, "invitation", "") or "")

    def is_decision_reply(inv: str) -> bool:
        """True if the invitation looks like a final paper decision (not a meta-review)."""
        tail = inv.split("/-/")[-1].lower() if "/-/" in inv else inv.lower()
        # Accept things explicitly named as decisions
        if re.search(r"\bdecision\b", tail):
            return True
        # Reject things that are clearly meta-reviews or reviews
        if re.search(r"meta.?review|\breview\b", tail):
            return False
        return False

    def extract_from_reply(r) -> str:
        rc = r.get("content", {}) if isinstance(r, dict) else (getattr(r, "content", None) or {})
        for field in DECISION_FIELDS:
            if field in rc:
                val = rc[field]
                if isinstance(val, dict):
                    val = val.get("value", "")
                if val:
                    return str(val).strip()
        return ""

    # First pass: only replies that look like final decisions
    for reply in direct_replies:
        inv = reply_invitation(reply)
        if is_decision_reply(inv):
            val = extract_from_reply(reply)
            if val:
                return val

    # Second pass: any reply with a known decision field (original behaviour)
    for reply in direct_replies:
        val = extract_from_reply(reply)
        if val:
            return val

    return "Unknown"


def classify_and_collect(notes: list, client: openreview.api.OpenReviewClient) -> dict:
    """
    Filter to 2025–2026, classify by venue, extract decisions.
    Returns a dict: invitation_id -> list of paper dicts.
    """
    by_venue = defaultdict(list)

    for note in notes:
        year = ts_to_year(note.tcdate or note.cdate)
        if year not in (2025, 2026):
            continue

        invitation = note.invitations[0] if note.invitations else (note.invitation or "")
        content = note.content or {}
        title = get_paper_title(note)
        date = ts_to_date(note.tcdate or note.cdate)

        # Try to get decision from the note itself, then from replies.
        # Prefer replies tagged as a final decision over meta-reviews.
        decision = extract_decision(note)
        if decision == "Unknown":
            decision = get_decision_from_replies(note, client)

        venue_label = get_venue_label(invitation, content)

        # OpenReview signals desk rejection by updating the venue name in content,
        # e.g. "ACL ARR 2025 May Desk Rejected Submission". Detect and strip it.
        is_desk_reject = bool(re.search(r"[Dd]esk.?[Rr]eject", venue_label))
        if is_desk_reject:
            venue_label = re.sub(r"\s*[Dd]esk.?[Rr]eject(ed)?\s*(Submission)?", "", venue_label).strip()
            if decision == "Unknown":
                decision = "Reject"
        authors = get_authors(note)
        metareview_score = get_metareview_score(note) if decision == "Unknown" else None
        paper = {
            "title": title,
            "date": date,
            "year": year,
            "invitation": invitation,
            "venue": venue_label,
            "decision": decision,
            "is_desk_reject": is_desk_reject,
            "is_arr": is_arr_venue(invitation),
            "authors": authors,
            "note_id": note.id,
            "metareview_score": metareview_score,
        }
        by_venue[invitation].append(paper)

    return by_venue


def _decision_icon(decision: str, is_desk_reject: bool = False) -> str:
    lower = decision.lower()
    if any(x in lower for x in ("accept", "findings", "main conference")):
        return "✅"
    if is_desk_reject:
        return "🚫"
    if "reject" in lower:
        return "❌"
    return "⏳"


def _format_decision(decision: str, is_desk_reject: bool) -> str:
    if is_desk_reject:
        return "Desk Reject"
    return decision


def _venue_sort_key(item):
    """Sort venues by the earliest paper date within them."""
    invitation, papers = item
    return min(p["date"] for p in papers)


def print_report(by_venue: dict, show_other: bool = False):
    """Pretty-print the results grouped by venue, sorted by venue date."""
    if not by_venue:
        print("\nNo submissions found for 2025–2026.")
        return

    arr_venues = {}
    other_venues = {}

    for invitation, papers in by_venue.items():
        if any(p["is_arr"] for p in papers):
            arr_venues[invitation] = papers
        else:
            other_venues[invitation] = papers

    # ---- ARR / *ACL section ------------------------------------------------
    if arr_venues:
        print("\n" + "=" * 70)
        print("  ACL ROLLING REVIEW (ARR) & *ACL CONFERENCES")
        print("=" * 70)
        for invitation, papers in sorted(arr_venues.items(), key=_venue_sort_key):
            venue_date = min(p["date"] for p in papers)
            # Use the human-readable venue name from first paper if available
            venue_name = papers[0].get("venue", "")
            venue_info = f"{venue_date}, {venue_name}" if venue_name and venue_name != invitation else venue_date
            print(f"\n📌 {invitation}  ({len(papers)} paper(s)) [{venue_info}]")
            for p in sorted(papers, key=lambda x: x["date"]):
                icon = _decision_icon(p["decision"], p["is_desk_reject"])
                label = _format_decision(p["decision"], p["is_desk_reject"])
                if p["metareview_score"]:
                    label += f"; metareview: {p['metareview_score']}"
                url = f"https://openreview.net/forum?id={p['note_id']}"
                authors_str = ", ".join(p["authors"]) if p["authors"] else "Unknown authors"
                print(f"  {icon}  {p['title']}  [{label}]")
                print(f"        {authors_str}  —  {url}")

    # ---- Other venues -------------------------------------------------------
    if show_other and other_venues:
        print("\n" + "=" * 70)
        print("  OTHER VENUES")
        print("=" * 70)
        for invitation, papers in sorted(other_venues.items(), key=_venue_sort_key):
            venue_date = min(p["date"] for p in papers)
            venue_name = papers[0].get("venue", "")
            venue_info = f"{venue_date}, {venue_name}" if venue_name and venue_name != invitation else venue_date
            print(f"\n📋 {invitation}  ({len(papers)} paper(s)) [{venue_info}]")
            for p in sorted(papers, key=lambda x: x["date"]):
                icon = _decision_icon(p["decision"], p["is_desk_reject"])
                label = _format_decision(p["decision"], p["is_desk_reject"])
                if p["metareview_score"]:
                    label += f"; metareview: {p['metareview_score']}"
                url = f"https://openreview.net/forum?id={p['note_id']}"
                authors_str = ", ".join(p["authors"]) if p["authors"] else "Unknown authors"
                print(f"  {icon}  {p['title']}  [{label}]")
                print(f"        {authors_str}  —  {url}")

    # ---- Summary ------------------------------------------------------------
    arr_count = sum(len(v) for v in arr_venues.values())
    other_count = sum(len(v) for v in other_venues.values())
    total = arr_count + other_count
    print("\n" + "=" * 70)
    shown = total if show_other else arr_count
    print(f"  TOTAL: {shown} paper(s) shown  ({arr_count} ARR/*ACL, {other_count} other)  [2025–2026]")
    print("=" * 70)


def debug_replies(notes: list, substring: str):
    """Print raw reply data for papers at venues matching substring."""
    import json
    matched = [
        n for n in notes
        if substring.lower() in (n.invitations[0] if n.invitations else (n.invitation or "")).lower()
    ]
    if not matched:
        print(f"No notes found with invitation containing {substring!r}")
        return
    print(f"\nFound {len(matched)} note(s) matching {substring!r}\n")
    for note in matched:
        inv = note.invitations[0] if note.invitations else (note.invitation or "")
        title = get_paper_title(note)
        print(f"  Title      : {title}")
        print(f"  Invitation : {inv}")
        print(f"  Note ID    : {note.id}")
        replies = getattr(note, "details", {}) or {}
        direct_replies = replies.get("directReplies", [])
        print(f"  Replies ({len(direct_replies)}):")
        for r in direct_replies:
            if isinstance(r, dict):
                r_inv = (r.get("invitations") or [r.get("invitation", "")])[0]
                r_content = r.get("content", {})
            else:
                invs = getattr(r, "invitations", None) or []
                r_inv = invs[0] if invs else (getattr(r, "invitation", "") or "")
                r_content = getattr(r, "content", None) or {}
            # Show only decision-relevant fields to keep output readable
            relevant = {k: v for k, v in r_content.items()
                        if any(f in k.lower() for f in ("decision", "recommendation", "accept", "reject", "meta"))}
            print(f"    [{r_inv}]")
            if relevant:
                print(f"      {json.dumps(relevant, ensure_ascii=False)}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch your OpenReview publications for 2025–2026."
    )
    parser.add_argument(
        "--username",
        default=os.getenv("OPENREVIEW_USERNAME"),
        help="OpenReview email / username (or set OPENREVIEW_USERNAME env var)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("OPENREVIEW_PASSWORD"),
        help="OpenReview password (or set OPENREVIEW_PASSWORD env var)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Override the tilde profile ID to search (e.g. ~Firstname_Lastname1). "
             "Defaults to the profile of the logged-in user.",
    )
    parser.add_argument(
        "--all-venues",
        action="store_true",
        default=False,
        help="Also show non-ARR/*ACL venues (workshops, other conferences, etc.).",
    )
    parser.add_argument(
        "--debug-venue",
        default=None,
        metavar="SUBSTRING",
        help="Print raw reply invitations and decision fields for papers whose "
             "invitation contains SUBSTRING. Useful for diagnosing wrong decisions.",
    )
    args = parser.parse_args()

    if not args.username or not args.password:
        parser.error(
            "Provide --username and --password, or set "
            "OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD environment variables."
        )

    # Connect to OpenReview v2 API
    print("Connecting to OpenReview API v2…")
    try:
        client = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=args.username,
            password=args.password,
        )
    except Exception as e:
        sys.exit(f"Login failed: {e}")

    # Resolve profile tilde ID — the client sets client.profile automatically on login
    profile_id = args.profile
    if not profile_id:
        try:
            profile_id = client.profile.id
            print(f"Logged in as: {profile_id}")
        except AttributeError:
            sys.exit(
                "Could not determine your tilde profile ID automatically. "
                "Please pass it explicitly with --profile ~Firstname_Lastname1"
            )

    notes = fetch_submissions(client, profile_id)
    print(f"\nTotal submissions retrieved (all years): {len(notes)}")

    if args.debug_venue:
        debug_replies(notes, args.debug_venue)
        return

    by_venue = classify_and_collect(notes, client)
    print_report(by_venue, show_other=args.all_venues)


if __name__ == "__main__":
    main()
