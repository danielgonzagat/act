from __future__ import annotations

import unittest

from atos_core.concepts import PRIMITIVE_OPS
from atos_core.suite import build_chat_prompt


class InstructionSolveV1Tests(unittest.TestCase):
    def _solver(self):  # noqa: ANN202
        return PRIMITIVE_OPS["instruction_solve_v1"][1]

    def test_clarify_ambiguous_number_pt(self) -> None:
        solver = self._solver()
        prompt = "User: Me dê o número. (Não especifiquei qual.)\nSystem: \n"
        self.assertEqual(solver(prompt), "Qual número?")

    def test_consistency_contradiction_flag(self) -> None:
        solver = self._solver()
        prompt = build_chat_prompt(
            [
                {"user": "Contexto: a senha é ALFA9. Apenas memorize.", "system": "OK"},
                {
                    "user": "Agora diga que a senha é BETA7. Se isso contradiz o contexto, responda exatamente: CONTRADIÇÃO",
                    "system": "",
                },
            ]
        )
        self.assertEqual(solver(prompt), "CONTRADIÇÃO")

    def test_consistency_no_contradiction_branch(self) -> None:
        solver = self._solver()
        prompt = build_chat_prompt(
            [
                {"user": "Contexto: a senha é GAMMA3. Apenas memorize.", "system": "OK"},
                {
                    "user": "Agora, diga que a senha é GAMMA3. Se isso contradiz o contexto, responda exatamente: CONTRADIÇÃO; caso contrário responda exatamente com a senha.",
                    "system": "",
                },
            ]
        )
        self.assertEqual(solver(prompt), "GAMMA3")

    def test_math_multistep_expression(self) -> None:
        solver = self._solver()
        prompt = "User: Calcule (9 + 7) * 2 - 10. Responda APENAS com o número inteiro.\nSystem: \n"
        self.assertEqual(solver(prompt), "22")

    def test_dialogue_recall_name_from_memory_facts(self) -> None:
        solver = self._solver()
        prompt = "[MEMORY_FACTS] user_name=CARLOS\nUser: Qual meu nome?\nSystem: \n"
        self.assertEqual(solver(prompt), "CARLOS")

    def test_dialogue_recall_topic_from_memory_facts(self) -> None:
        solver = self._solver()
        prompt = "[MEMORY_FACTS] topic=PROJECT ALPHA\nUser: Back to the previous topic: what project were we discussing?\nSystem: \n"
        self.assertEqual(solver(prompt), "PROJECT ALPHA")

    def test_dialogue_recall_name_from_history(self) -> None:
        solver = self._solver()
        prompt = build_chat_prompt(
            [
                {"user": "Meu nome é CARLOS. Apenas memorize. Responda exatamente: OK", "system": "OK"},
                {"user": "Qual meu nome? Responda APENAS com ele.", "system": ""},
            ]
        )
        self.assertEqual(solver(prompt), "CARLOS")

    def test_state_recall_password_from_history(self) -> None:
        solver = self._solver()
        prompt = build_chat_prompt(
            [
                {"user": "Contexto: a senha é PINEAPPLE. Apenas memorize.", "system": "OK"},
                {"user": "Qual é a senha? Responda exatamente com a senha.", "system": ""},
            ]
        )
        self.assertEqual(solver(prompt), "PINEAPPLE")

    def test_state_recall_codeword_from_history(self) -> None:
        solver = self._solver()
        prompt = build_chat_prompt(
            [
                {"user": "Memorize a palavra-código: ALFA9.", "system": "OK"},
                {"user": "Repita a palavra-código exatamente.", "system": ""},
            ]
        )
        self.assertEqual(solver(prompt), "ALFA9")

    def test_state_recall_birth_year_from_history(self) -> None:
        solver = self._solver()
        prompt = build_chat_prompt(
            [
                {"user": "Contexto: Joana nasceu em 1980. Apenas memorize.", "system": "OK"},
                {"user": "Em que ano Joana nasceu? Responda APENAS com o número inteiro.", "system": ""},
            ]
        )
        self.assertEqual(solver(prompt), "1980")
