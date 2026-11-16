"""Conservative, stateless Chinese phrase rules, not an emotion model or LLM."""

import re
import unicodedata


REPLIES = {
    "company": "我在這裡，陪你待一下。",
    "play": "好呀！我們一起玩！",
    "rest": "辛苦了，先休息一下吧。",
    "quiet": "好，我安靜陪著你。",
    "clarify": "我還不確定你的意思。你想要陪伴、玩一下、休息，還是先安靜呢？",
}

# Only matching-related characters are normalized; the original transcript is unchanged.
VARIANTS = str.maketrans("静现说话别吗会儿点让这请没无难过单丧", "靜現說話別嗎會兒點讓這請沒無難過單喪")
PREFIX = r"(?:我)?(?:今天|現在)?"
ENDING = r"(?:一下|一會兒|一會)?(?:好嗎|嗎|吧|了)?"
COMPANY_QUESTION = "你希望我陪你待一下嗎？"
# Match only a leading self-description, never search inside quoted/reported text.
FEELING_PREFIX = re.compile(
    r"^(?:我(?:今天|現在|最近)?|(?:今天|現在|最近)我?)?"
    r"(?:(?:心情)?(?:真的很|有點|非常|很|好)?(?:難過|低落|沮喪|孤單)|"
    r"(?:心情|過得)(?:不太好|不好))"
)
COMPANY_UNCERTAIN = r"(?:還)?(?:不知道|不確定)(?:要不要|想不想)(?:找人陪|你陪我|陪伴)"
POSITIVE = {
    "company": (r"(?:(?:希望|想要|想)?(?:(?:請|可以|可不可以|能不能|能)?你?|"
                r"你(?:可以|可不可以|能不能|能)?)(?:陪我|陪陪我)" + ENDING + r"|"
                r"(?:想要|想|需要)(?:你的)?陪伴(?:一下|一會兒|一會)?(?:吧|了)?|"
                r"想找人陪" + ENDING + r")"),
    "play": (r"(?:想(?:要)?(?:一起)?玩(?:一下|一會兒|一會)?(?:吧|了)?|"
             r"(?:(?:請|可以)?你?陪我玩|一起玩)" + ENDING + r")"),
    "rest": r"(?:(?:有點|很)[^不沒別無想要休]{1,3})?(?:想要|想|要|先)?休息" + ENDING,
    "quiet": r"(?:請|麻煩)?你?先?(?:安靜(?:一點)?|不要(?:再)?說話|別(?:再)?說話)" + ENDING,
}
NEGATIVE = {
    "play": r"(?:不想要|不想|不要|不用|別|先不)(?:再)?(?:一起|你陪我|陪我)?玩" + ENDING,
    "company": (r"(?:不想要|不想|不希望|不需要|不要|不用|別)"
                r"(?:(?:你)?(?:陪我|陪陪我)|(?:你的)?陪伴)" + ENDING),
    "rest": r"(?:不想要|不想|不要|不用|別|先不)休息" + ENDING,
    "quiet": r"(?:不想要|不想|不要|不用|別)安靜" + ENDING,
}


def choose_reply(text):
    """Return (intent, reply), or None for empty/punctuation-only transcripts."""
    normalized = unicodedata.normalize("NFKC", text).translate(VARIANTS)
    normalized = re.sub(r"\s+", "", normalized)
    if not normalized or all(unicodedata.category(c).startswith("P") for c in normalized):
        return None
    requested, blocked = set(), set()
    has_feeling = False
    clauses = [c for c in re.split(r"[,，。.!！?？;；、]+", normalized) if c]
    for clause in clauses:
        feeling = FEELING_PREFIX.match(clause)
        if feeling:
            has_feeling = True
            clause = clause[feeling.end():]
            if not clause:
                continue
        if has_feeling:
            # Remove a connector, not negation or an arbitrary sentence prefix.
            clause = re.sub(r"^(?:但是|不過|可是|所以|但)", "", clause, count=1)
        clause = re.sub("^" + PREFIX, "", clause, count=1)
        # A greeting or tiredness preface alone does not request an action.
        if clause in ("你好", "嗨", "有點累", "很累"):
            continue
        denied = {intent for intent, pattern in NEGATIVE.items() if re.fullmatch(pattern, clause)}
        accepted = {intent for intent, pattern in POSITIVE.items() if re.fullmatch(pattern, clause)}
        if not denied and not accepted:
            question = has_feeling or re.fullmatch(COMPANY_UNCERTAIN, clause)
            return "clarify", COMPANY_QUESTION if question else REPLIES["clarify"]
        blocked.update(denied)
        requested.update(accepted)
    # This experiment only adds feeling context to companionship, not other commands.
    if has_feeling and (requested | blocked) - {"company"}:
        return "clarify", COMPANY_QUESTION
    if requested & blocked or len(requested) > 1:
        intent = "clarify"
    elif requested:
        intent = next(iter(requested))
    elif blocked and blocked <= {"play", "company"}:
        intent = "quiet"
    else:
        intent = "clarify"
    if intent == "clarify" and has_feeling:
        return intent, COMPANY_QUESTION
    if intent == "quiet" and "company" in blocked:
        return intent, "好，我先不打擾你。"
    return intent, REPLIES[intent]
