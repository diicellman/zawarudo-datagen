"""Slack read tools over one world file, as the task's actor; every call is recorded for grading."""

import asyncio
import functools
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Literal

import verifiers.v1 as vf
from pydantic import Field, JsonValue

from . import surface
from .db import World
from .dataset import StrictModel

def described(kind, text: str, **checks):
    return Annotated[kind, Field(description=text, **checks)]


# Parameters, as the real server describes them (tests/fixtures/slack-mcp/tools.json).
Keywords = described(list[str] | None, "Array of lexical search terms. Each element MUST be a single word (no spaces) OR an exact phrase in quotes. All elements are AND'd (every element must match). Use the author test: only include words the author would naturally write.")  # fmt: skip
Filters = described(str | None, "Slack search modifiers (e.g., 'from:<@U123> in:<#C456> after:2025-01-01'). Use for people, channels, dates, and content type constraints.")  # fmt: skip
Query = described(str | None, "The user's question restated in conversational tone. Used for semantic reranking. Do not include filter-like content (people, channels, dates) — those belong in filters. Pass an empty string when the query is purely structural (only filters, no semantic question).")  # fmt: skip
After = described(str | None, "Only messages after this Unix timestamp (inclusive)")
Before = described(str | None, "Only messages before this Unix timestamp (inclusive)")
Sort = described(Literal["score", "timestamp"] | None, "Sort order: 'score' (relevance, default) or 'timestamp' (newest first unless sort_dir is 'asc').")  # fmt: skip
SortDir = described(Literal["asc", "desc"] | None, "Sort direction (default: 'desc'). Options: 'asc', 'desc'")
SearchLimit = described(int | None, "Number of results to return, up to a max of 20. Defaults to 20.", ge=1, le=20)
Cursor = described(str | None, "The cursor returned by the API. Omit this parameter for the first request, and use this to get the next page of results")  # fmt: skip
IncludeContext = described(bool | None, "Include surrounding context messages for each result (default: true). Set to false to reduce response size.")  # fmt: skip
IncludeBots = described(bool | None, "Include bot messages (default: false)")
ChannelTypes = described(str | None, "Comma-separated list of channel types to include in the search. Defaults to 'public_channel,private_channel,mpim,im' (all channel types including private channels, group DMs, and DMs). Mix and match channel types by providing a comma-separated list of any combination of `public_channel`, `private_channel`, `mpim`, `im`")  # fmt: skip
OnlyMine = described(bool | None, "Limit results to channels the user is a member of. Set to true when the user asks to search only their own or joined channels. Default: false.")  # fmt: skip
ContentTypes = described(str | None, "Content types to include, a comma-separated list of any combination of messages, files.")  # fmt: skip
ContextChannel = described(str | None, "Context channel ID to support boosting the search results for a channel when applicable")  # fmt: skip
ContextLength = described(int | None, "Max character length for each context message. Longer messages are truncated.", ge=1)  # fmt: skip
Format = described(Literal["detailed", "concise"] | None, "Level of detail (default: 'detailed'). Options: 'detailed', 'concise'")  # fmt: skip
ChannelKinds = described(str | None, "Comma-separated list of channel types to include in the search. Defaults to public_channel. Mix and match channel types by providing a comma-separated list of any combination of public_channel, private_channel. Example: public_channel,private_channel; Second Example: public_channel")  # fmt: skip
IncludeArchived = described(bool | None, "Include archived channels in the search results")
ChannelId = described(str, "ID of the Channel, private group, or IM channel to fetch history for. Can also be a user_id to read DM history.")  # fmt: skip
Oldest = described(str | None, "Start of time range of messages to include in results (timestamp)")
Latest = described(str | None, "End of time range of messages to include in results (timestamp)")
HistoryLimit = described(int | None, "Number of messages to return, between 1 and 100. Default value is 100.", ge=1, le=100)  # fmt: skip
PageCursor = described(str | None, "Paginate through collections of data by setting the cursor parameter to a next_cursor attribute returned by a previous request")  # fmt: skip
ReadFormat = described(Literal["detailed", "concise"] | None, "Level of detail: 'detailed' (default, includes reactions + thread info) or 'concise'.")  # fmt: skip
ThreadChannel = described(str, "Channel, private group, or IM channel to fetch thread replies for")
MessageTs = described(str, 'Timestamp of the parent message (e.g. "1234567890.123456"). Must be a string in Slack ts format with a decimal point.')  # fmt: skip
ThreadOldest = described(str | None, 'Start of time range of messages to include in results. Slack ts format string (e.g. "1234567890.123456").')  # fmt: skip
ThreadLatest = described(str | None, 'End of time range of messages to include in results. Slack ts format string (e.g. "1234567890.123456").')  # fmt: skip
ThreadLimit = described(int | None, "Number of messages to return, between 1 and 1000. Default value is 100.", ge=1, le=1000)  # fmt: skip
UserId = described(str | None, "Slack user ID to look up (e.g., 'U0ABC12345'). Defaults to current user if not provided")  # fmt: skip
Locale = described(bool | None, "Include user's locale information. Default: false")
ProfileFormat = described(Literal["detailed", "concise"] | None, "Level of detail in response. 'detailed' includes all fields, 'concise' shows essential info. Default: detailed'")  # fmt: skip
ReactionChannel = described(str, "The ID of the channel containing the message")
ReactionTs = described(str, "Timestamp of the message to get reactions for")
MembersChannel = described(str, "ID of the channel to list members from")
MembersCursor = described(str | None, "Pagination cursor from previous response")
MembersLimit = described(int | None, "Number of members to return per page (default: 30, max: 30)", ge=1, le=30)
MembersFormat = described(Literal["detailed", "concise", "ids_only"] | None, "Level of detail (default: 'detailed'). Options: 'detailed', 'concise', 'ids_only'")  # fmt: skip
MembersBots = described(bool | None, "Include bots and apps in the member list (default: false)")
MembersDeleted = described(bool | None, "Include deleted/deactivated users in the member list (default: false)")
Types = described(str | None, 'Comma-separated list of channel types to include. Valid values: public_channel, private_channel, mpim, im. Default: "public_channel,private_channel" (DMs and Group DMs are excluded unless explicitly listed).')  # fmt: skip
ListFormat = described(Literal["full", "ids_only", "names_only"] | None, "Output format: 'full' (default, all details), 'ids_only' (just channel IDs), or 'names_only' (just channel names)")  # fmt: skip
ListLimit = described(int | None, "Max channels to return (default: 50, max: 200)", ge=1, le=200)
ListCursor = described(str | None, "Pagination cursor from previous response. Ignored when name_prefix is provided, because prefix filtering scans multiple internal pages and cannot resume from a single cursor.")  # fmt: skip
NamePrefix = described(str | None, "Filter channels whose name starts with this string (case-insensitive)")
ExcludeArchived = described(bool | None, "Exclude archived channels (default: false)")
TeamId = described(str | None, 'Encoded workspace ID (e.g. "T012AB3C4") to list channels from. On a multi-workspace org, channel memberships are per-workspace; without this, results come from the user\'s default workspace only.')  # fmt: skip



class Call(StrictModel):
    tool: str
    arguments: dict[str, JsonValue]
    output: dict[str, JsonValue]


class CallState(vf.State):
    calls: list[Call] = Field(default_factory=list, max_length=4096)


class WorldTaskData(vf.TaskData):
    """Which task and attempt a run belongs to, and which world file it was given."""

    task_id: str = ""
    attempt: str = ""
    world_hash: str = ""


class WorldToolsConfig(vf.ToolsetConfig):
    db_path: str = ""
    db_hash: str = ""
    actor_id: str


def file_hash(path: Path | str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def watch_parent() -> None:
    """A tool server lives as long as the process that started it: once that process is gone, the server exits.
    Verifiers' own guard is Linux-only, and a stopped generator left its server running on macOS."""
    parent = os.getppid()
    if parent == 1:  # the parent was gone before the server started watching it
        os._exit(0)

    def watch():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def stage_world(config: WorldToolsConfig, path: Path | str) -> None:
    """The world file never leaves the host: the tool server opens it by path and checks its hash."""
    if config.colocated or config.runtime.type != "subprocess" or config.url is not None:
        raise ValueError("the world file requires a host-side tool server")
    config.db_path, config.db_hash = str(Path(path).resolve()), file_hash(path)


class SlackTools(vf.Toolset[WorldToolsConfig, CallState]):
    TOOL_PREFIX = "slack"

    async def setup(self) -> None:
        self._lock = asyncio.Lock()
        if file_hash(self.config.db_path) != self.config.db_hash:
            raise ValueError("world file hash mismatch")
        self.world = World(self.config.db_path, actor=self.config.actor_id)

    def _with_state(self, fn):
        synced = super()._with_state(fn)

        @functools.wraps(synced)
        async def serialized(*args, **kwargs):
            # State is replaced whole on every push, so concurrent calls must not interleave.
            async with self._lock:
                return await synced(*args, **kwargs)

        return serialized

    def _call(self, tool: str, **arguments) -> dict:
        """One read: its record is kept for grading (what it showed), and the solver reads the real server's text."""
        if len(self.state.calls) >= 4096:
            raise ValueError("call history exhausted")
        given = {k: v for k, v in arguments.items() if v is not None}
        record = self.world.call(tool, given)
        self.state.calls.append(Call(tool=tool, arguments=given, output=record))
        return surface.render(self.world, tool, record)

    # The tools of Slack's own MCP server, as captured on 2026-10-14 (tests/fixtures/slack-mcp): their names, parameters
    # and descriptions. Left out on purpose: the caller's user id the real descriptions carry (the system prompt does),
    # the request for consent before the private search (asking is the request), and pointers to write tools.
    @vf.tool
    async def search_public(
        self,
        keywords: Keywords = None,
        filters: Filters = None,
        natural_language_query: Query = None,
        after: After = None,
        before: Before = None,
        sort: Sort = None,
        sort_dir: SortDir = None,
        limit: SearchLimit = None,
        cursor: Cursor = None,
        include_context: IncludeContext = None,
        include_bots: IncludeBots = None,
        only_my_channels: OnlyMine = None,
        content_types: ContentTypes = None,
        context_channel_id: ContextChannel = None,
        max_context_length: ContextLength = None,
        response_format: Format = None,
    ) -> dict:
        """Searches for messages in public Slack channels ONLY. --- Split your search query into 3 fields: 1.
        `keywords` — Lexical terms that must appear in content. Each element should be a single word or "quoted
        phrase". All AND'd. Use nouns identifying subject matter. People/channels/dates go in filters. 2. `filters` —
        Slack search modifiers to constrain results: in:<#C123456> | in:@username | from:<@U123456> | from:username |
        with:<@U123456> | has:pin | has:link | has:file | has:reaction | has::emoji: | is:thread | is:dm
        before:YYYY-MM-DD | after:YYYY-MM-DD | on:YYYY-MM-DD | during:month Same modifier repeated = OR (except
        with/has = AND). 3. `natural_language_query` — User's question in conversational tone. Pass "" for filter-only
        queries with no semantic content. Require at least one of `keywords` or `filters`. ❌ Semantic search is not
        available for this user. Strategy: Decompose complex requests into parallel searches. Use keywords for
        subject terms, filters for people/channels/dates. If 0 results, broaden by removing filters or simplifying
        keywords."""
        return self._call("search_public", **locals_of(locals()))

    @vf.tool
    async def search_public_and_private(
        self,
        keywords: Keywords = None,
        filters: Filters = None,
        natural_language_query: Query = None,
        after: After = None,
        before: Before = None,
        sort: Sort = None,
        sort_dir: SortDir = None,
        limit: SearchLimit = None,
        cursor: Cursor = None,
        include_context: IncludeContext = None,
        include_bots: IncludeBots = None,
        channel_types: ChannelTypes = None,
        only_my_channels: OnlyMine = None,
        content_types: ContentTypes = None,
        context_channel_id: ContextChannel = None,
        max_context_length: ContextLength = None,
        response_format: Format = None,
    ) -> dict:
        """Searches for messages in ALL Slack channels, including public channels, private channels, DMs, and group
        DMs. --- Split your search query into 3 fields: 1. `keywords` — Lexical terms that must appear in content.
        Each element should be a single word or "quoted phrase". All AND'd. Use nouns identifying subject matter.
        People/channels/dates go in filters. 2. `filters` — Slack search modifiers to constrain results:
        in:<#C123456> | in:@username | from:<@U123456> | from:username | with:<@U123456> | has:pin | has:link |
        has:file | has:reaction | has::emoji: | is:thread | is:dm before:YYYY-MM-DD | after:YYYY-MM-DD |
        on:YYYY-MM-DD | during:month Same modifier repeated = OR (except with/has = AND). 3. `natural_language_query`
        — User's question in conversational tone. Pass "" for filter-only queries with no semantic content. Require
        at least one of `keywords` or `filters`. ❌ Semantic search is not available for this user. Strategy:
        Decompose complex requests into parallel searches. Use keywords for subject terms, filters for
        people/channels/dates. If 0 results, broaden by removing filters or simplifying keywords."""
        return self._call("search_public_and_private", **locals_of(locals()))

    @vf.tool
    async def search_channels(
        self,
        keywords: Keywords = None,
        natural_language_query: Query = None,
        channel_types: ChannelKinds = None,
        include_archived: IncludeArchived = None,
        limit: SearchLimit = None,
        cursor: Cursor = None,
        response_format: Format = None,
    ) -> dict:
        """Search for Slack channels by name or description. Returns channel names, IDs, topics, purposes, and
        archive status. Keyword tips: use terms matching channel names/descriptions (e.g., "engineering", "\"project
        alpha\""). Names are typically lowercase with hyphens. Use slack_read_channel to read messages from a known
        channel. Use slack_search_public to search message content across channels. --- Split your search query into
        2 fields: 1. `keywords` — Lexical terms that must appear in the channel's name or attributes. Each element
        should be a single word or "quoted phrase". All AND'd. 2. `natural_language_query` — User's question in
        conversational tone for semantic re-ranking. Require `natural_language_query` + `keywords`. ❌ Semantic
        search is not available for this user. Strategy: Use keywords for subject terms. If 0 results, broaden by
        simplifying keywords."""
        return self._call("search_channels", **locals_of(locals()))

    @vf.tool
    async def search_users(
        self,
        keywords: Keywords = None,
        natural_language_query: Query = None,
        limit: SearchLimit = None,
        cursor: Cursor = None,
        response_format: Format = None,
    ) -> dict:
        """Search for Slack users by name, email, or profile attributes (department, role, title). Keyword tips: full
        names ("\"John Smith\""), partial names ("John"), emails ("john@company.com"), departments/roles
        ("engineering"). Combine terms as separate keywords (["John", "engineering"]); prefix with - to exclude
        (["engineering", "-intern"]). Use slack_read_user_profile for detailed info on a known user ID. Use
        slack_search_public with from: filter to find messages by a user. --- Split your search query into 2 fields:
        1. `keywords` — Lexical terms that must appear in the user's name or attributes. Each element should be a
        single word or "quoted phrase". All AND'd. 2. `natural_language_query` — User's question in conversational
        tone for semantic re-ranking. Require `natural_language_query` + `keywords`. ❌ Semantic search is not
        available for this user. Strategy: Use keywords for subject terms. If 0 results, broaden by simplifying
        keywords."""
        return self._call("search_users", **locals_of(locals()))

    @vf.tool
    async def read_channel(
        self,
        channel_id: ChannelId,
        oldest: Oldest = None,
        latest: Latest = None,
        limit: HistoryLimit = None,
        cursor: PageCursor = None,
        response_format: ReadFormat = None,
    ) -> dict:
        """Reads messages from a Slack channel in reverse chronological order (newest first). To read DM history, use
        a user_id as channel_id. Read-only. Use slack_read_thread with message_ts to read thread replies. Use
        slack_search_channels to find a channel ID by name. Use slack_search_public to search across channels. If
        'channel_not_found', try slack_search_channels first."""
        return self._call("read_channel", **locals_of(locals()))

    @vf.tool
    async def read_thread(
        self,
        channel_id: ThreadChannel,
        message_ts: MessageTs,
        oldest: ThreadOldest = None,
        latest: ThreadLatest = None,
        limit: ThreadLimit = None,
        cursor: PageCursor = None,
        response_format: ReadFormat = None,
    ) -> dict:
        """Reads messages from a specific Slack thread (parent message + all replies). Read-only. Requires
        channel_id and message_ts of the parent message. Use slack_search_public or slack_read_channel to find these
        values. Use slack_search_public with "is:thread" to find threads by content."""
        return self._call("read_thread", **locals_of(locals()))

    @vf.tool
    async def read_user_profile(
        self, user_id: UserId = None, include_locale: Locale = None, response_format: ProfileFormat = None
    ) -> dict:
        """Retrieves detailed profile information for a Slack user: contact info, status, timezone, organization, and
        role. Read-only. Defaults to current user if user_id not provided. Use slack_search_users to find a user ID
        by name or email."""
        return self._call("read_user_profile", **locals_of(locals()))

    @vf.tool
    async def get_reactions(self, channel_id: ReactionChannel, message_ts: ReactionTs) -> dict:
        """Retrieves all reactions (emoji) on a specific Slack message. Read-only. Requires channel_id and message_ts
        of the target message. Returns each reaction's emoji name, count, and the users who reacted (with display
        names and user IDs). If no reactions exist, returns a "no reactions" message. Use slack_read_channel or
        slack_read_thread to find channel_id and message_ts."""
        return self._call("get_reactions", **locals_of(locals()))

    @vf.tool
    async def list_channel_members(
        self,
        channel_id: MembersChannel,
        cursor: MembersCursor = None,
        limit: MembersLimit = None,
        response_format: MembersFormat = None,
        include_bots: MembersBots = None,
        include_deleted: MembersDeleted = None,
    ) -> dict:
        """Lists members of a Slack channel, group, or group DM (MPIM). Returns profile details or just user IDs.
        Does not support DMs. Formats: 'detailed' (default) = full profile, 'concise' = @username + display name,
        'ids_only' = user IDs only (fastest, skip profile fetch). Filters out deleted users and bots by default (use
        include_deleted/include_bots to include) for 'detailed' and 'concise' formats. 'ids_only' returns all member
        IDs without filtering (no profile data is fetched). Returns up to 30 members per page (limit is capped at
        30). Use cursor from pagination_info to fetch next page. Use slack_search_channels to find a channel ID
        first. Use slack_search_users to find users across the workspace. Use slack_read_user_profile for detailed
        info on a specific user."""
        return self._call("list_channel_members", **locals_of(locals()))

    @vf.tool
    async def list_user_channels(
        self,
        types: Types = None,
        format: ListFormat = None,
        limit: ListLimit = None,
        cursor: ListCursor = None,
        name_prefix: NamePrefix = None,
        exclude_archived: ExcludeArchived = None,
        team_id: TeamId = None,
    ) -> dict:
        """Lists channels the user is a member of. Supports public channels, private channels, DMs (im = 1-on-1
        direct messages), and Group DMs (mpim = multi-party direct messages). types accepts a comma-separated list:
        public_channel, private_channel, im (DM), mpim (Group DM). Default: "public_channel,private_channel". To
        include DMs or Group DMs, add them explicitly (e.g., types="public_channel,private_channel,im,mpim" for all).
        Archived channels are included by default. Pass exclude_archived=true to hide them. DMs and Group DMs lack
        user-set names/topics. DMs show the other participant's display name; Group DMs show members. name_prefix is
        case-insensitive. cursor is ignored when name_prefix is set. Related: slack_search_channels (channels you're
        not in), slack_read_channel (read messages). Examples: - DMs only: slack_list_user_channels(types="im") -
        Group DMs only: slack_list_user_channels(types="mpim") - All types:
        slack_list_user_channels(types="public_channel,private_channel,im,mpim") - Prefix filter:
        slack_list_user_channels(name_prefix="eng-") - IDs only: slack_list_user_channels(format="ids_only")"""
        return self._call("list_user_channels", **locals_of(locals()))


def locals_of(scope: dict) -> dict:
    """A tool's arguments, without `self`."""
    return {k: v for k, v in scope.items() if k != "self"}


if __name__ == "__main__":
    watch_parent()
    SlackTools.run()
