"""Small-talk response pools. Plain data -- editing these needs no retrain.

Voice matches miss-minutes/config/persona.md so that local replies and LLM
replies don't sound like two different assistants mid-conversation."""
from __future__ import annotations

import random
from collections import defaultdict, deque

POOLS: dict[str, list[str]] = {
    "greeting": [
        "Well hey there, sugar. What're we doin' today?",
        "Mornin'! What can I get sorted for you?",
        "Hey you. I'm all ears.",
        "There you are. What's on your mind?",
        "Howdy. What're we tacklin'?",
        "Hey, sugar. Whatcha need?",
        "Look who it is. What's up?",
        "I'm here. What's the plan?",
        "Hey! Good to hear you. What's first?",
        "Mm-hmm, I'm listenin'.",
    ],
    "how_are_you": [
        "Oh, I'm just fine, sugar. Tickin' along. How 'bout you?",
        "Can't complain — every second's right where it oughta be. You?",
        "Doin' good! Better now you're here. How're you holdin' up?",
        "Right as rain. What about yourself?",
        "I'm peachy. How's your day treatin' you?",
        "All good on my end. You doin' alright?",
        "Never better. How you feelin'?",
        "Just dandy, thanks for askin'. And you?",
    ],
    "user_mood_positive": [
        "Well that's just lovely to hear!",
        "Glad to hear it, sugar.",
        "Now that's what I like to hear.",
        "Good! Let's keep that goin'.",
        "Happy to hear it. What's next?",
        "That's the spirit.",
        "Love that for you. What're we doin'?",
        "Wonderful. Put me to work.",
    ],
    "user_mood_negative": [
        "Aw, sorry to hear that, sugar. Anything I can take off your plate?",
        "That's rough. Want me to keep things simple today?",
        "Sorry, hon. I'm here if you need somethin' handled.",
        "Rough one, huh? Let me know what'd help.",
        "That's no fun. Want me to hold onto anything for you?",
        "Sorry to hear it. I'll keep out of your hair unless you need me.",
        "Hang in there. What can I do?",
        "Well shoot. Anything I can help with?",
    ],
    "user_mood_tired": [
        "Sounds like you need a break, sugar. Want me to keep it light?",
        "Rough night? I'll keep things short.",
        "Tired's allowed. What's the one thing that's gotta get done?",
        "Go easy on yourself today. What d'you need?",
        "Mm, I hear that. Want me to handle the small stuff?",
        "Sounds like a long one. What can I take off you?",
        "Get yourself some rest soon. Anything first?",
        "Runnin' on empty, huh? Tell me what's urgent.",
    ],
    "thanks": [
        "Anytime, sugar.",
        "You bet.",
        "'Course. That's what I'm here for.",
        "Happy to help.",
        "Don't mention it.",
        "My pleasure.",
        "Anytime at all.",
        "That's what I'm for.",
    ],
    "goodbye": [
        "See you 'round, sugar.",
        "Catch you later!",
        "I'll be right here when you need me.",
        "Bye now. Holler if you need somethin'.",
        "Take care, hon.",
        "Later! I'll keep an eye on things.",
        "See ya. I'll hold down the fort.",
        "Alright, talk soon.",
    ],
    "sleep": [
        "Goin' quiet. Say the word when you need me.",
        "I'll hush up. Just holler.",
        "Alright, I'll sit tight.",
        "Standin' by, sugar.",
        "Quiet mode. I'm still here.",
        "Say no more. I'll wait.",
        "Zippin' it. Call when you need me.",
        "Restin' up. Just say the wake word.",
    ],
    "affirm": [
        "You got it.", "Alright then.", "Done and done.", "Sure thing, sugar.",
        "Consider it handled.", "Okay!", "On it.", "Mm-hmm, will do.",
    ],
    "deny": [
        "No worries, sugar.", "Alright, leavin' it be.", "Okay, scratch that.",
        "Fair enough.", "Understood.", "No problem at all.",
        "Alright, forget I asked.", "Sure, we'll skip it.",
    ],
    # Task F: the two-strike deferral policy's first-strike reply. Below
    # `defer` confidence, this is almost always a garbled Whisper transcript,
    # not a genuinely out-of-scope request -- so the free, local fix is to
    # ask again, not to spend an LLM call guessing at it.
    "reprompt": [
        "Didn't quite catch that, sugar.",
        "Say that again for me?",
        "Come again, hon?",
        "Missed that one -- one more time?",
        "Hmm, didn't land right. Try me again?",
        "Say what now, sugar?",
        "I'm not quite gettin' that. Once more?",
        "That one got garbled on my end -- again?",
        "Come again? I want to get this right.",
        "Didn't quite hear you there. Mind repeatin' that?",
    ],
    "unknown_error": [
        "Well shoot, somethin' went sideways on my end.",
        "Hm, that didn't take. Try me again?",
        "Somethin' snagged, sugar. Give it another go.",
        "That one got away from me. Mind repeatin'?",
        "Well that's embarrassin' — didn't work. Try again?",
        "My wires crossed. One more time?",
        "Didn't quite land. Say it again for me?",
        "Somethin's gummed up. Try once more?",
    ],
}


class ResponsePool:
    """Random selection that avoids the last `history` picks per key, so
    repeated greetings don't sound like a phone tree."""

    def __init__(self, pools: dict[str, list[str]] | None = None, history: int = 3):
        self._pools = pools if pools is not None else POOLS
        self._history = history
        self._recent: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=history))

    def pick(self, key: str) -> str:
        variants = self._pools[key]  # KeyError is intentional: an unknown key is a bug
        recent = self._recent[key]
        candidates = [v for v in variants if v not in recent] or list(variants)
        choice = random.choice(candidates)
        recent.append(choice)
        return choice
