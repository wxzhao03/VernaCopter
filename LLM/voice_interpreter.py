"""
Voice utterance interpreter for in-flight drone commands.
Classifies a Whisper transcription as: yes / no / goal_name.
GPT only sees the available goal names — never obstacle names.
"""

import os
import re
from LLM.GPT import GPT


class VoiceInterpreter:
    """
    Single-shot GPT classifier for voice utterances during simulation.

    Parameters
    ----------
    goal_names : list[str]
        Names of reachable goal objects (e.g. ['goal1', 'goal2']).
    GPT_model : str
        OpenAI model to use.
    """

    _YES_WORDS  = {'yes', 'yeah', 'sure', 'okay', 'yep', 'accept', 'confirmed'}
    _NO_WORDS   = {'no', 'nope', 'reject', 'cancel', 'deny'}
    _WORD_TO_DIGIT = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    }

    def __init__(self, goal_names, GPT_model="gpt-4o"):
        self.goal_names = [g.lower() for g in goal_names]
        self.gpt = GPT(GPT_model)
        self._system_prompt = self._load_instructions()

    def _load_instructions(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_path, 'instructions', 'voice_interpreter_instructions.txt')
        with open(path, 'r', encoding='utf-8') as f:
            instructions = f.read()
        goals_str = ', '.join(self.goal_names)
        return instructions.replace('GOALS', goals_str)

    def interpret(self, text: str) -> str | None:
        """
        Classify *text* and return one of: 'yes', 'no', a goal name, or None.

        Fast keyword pre-check avoids a GPT round-trip for obvious yes/no.
        Falls back to GPT for ambiguous or goal-bearing utterances.

        Returns
        -------
        str | None
            'yes', 'no', goal name (lowercase), or None if unrecognised.
        """
        # Normalize number words → digits ("three" → "3"), then collapse "goal 3" → "goal3"
        text_norm = text.lower().strip()
        for word, digit in self._WORD_TO_DIGIT.items():
            text_norm = re.sub(rf'\b{word}\b', digit, text_norm)
        text_norm = re.sub(r'([a-zA-Z])\s+(\d)', r'\1\2', text_norm)
        words = set(text_norm.split())

        # Fast path: unambiguous yes/no
        if words & self._YES_WORDS and not (words & self._NO_WORDS):
            return 'yes'
        if words & self._NO_WORDS and not (words & self._YES_WORDS):
            return 'no'

        # Fast path: goal name substring match on normalized text
        for g in self.goal_names:
            if g in text_norm:
                return g

        # GPT fallback for phrasing like "head to the top-right area" or "go to the left goal"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": text},
        ]
        try:
            response = self.gpt.chatcompletion(messages).strip().lower()
        except Exception as e:
            print(f"[VoiceInterpreter] GPT error: {e}")
            return None

        if response in ('yes', 'no'):
            return response
        for g in self.goal_names:
            if g in response:
                return g

        print(f"[VoiceInterpreter] Unrecognised response: '{response}'")
        return None
