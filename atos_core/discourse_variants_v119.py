from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex


def _norm_ws(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def _tokenize(text: str) -> List[str]:
    t = _norm_ws(text).lower()
    if not t:
        return []
    return t.split()


def prefix2_from_text_v119(text: str) -> str:
    toks = _tokenize(text)
    if not toks:
        return ""
    return " ".join(toks[:2])


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _stable_hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


ACK_BANK_V119: List[str] = [
    "Certo.",
    "Beleza.",
    "Ok.",
    "Entendi.",
    "Perfeito.",
    "Tudo bem.",
    "Combinado.",
    "Fechado.",
    "Ótimo.",
    "Certo — seguindo.",
    "Beleza — seguindo.",
    "Ok — seguindo.",
    "Entendi — vou continuar.",
    "Perfeito — vou continuar.",
    "Tudo bem — vou seguir.",
]

OPTIONS_PREAMBLE_BANK_V119: List[str] = [
    "Aqui estão as opções:",
    "Estas são as opções:",
    "Veja as opções:",
    "Opções disponíveis:",
    "Tenho estas opções:",
    "Posso seguir por estas opções:",
    "Sugestões de caminho (A/B/C):",
    "Alternativas (A/B/C):",
    "Caminhos possíveis:",
    "Vou listar as opções:",
    "Para seguir, estas são as opções:",
    "Para avançar, veja as opções:",
    "Para decidir, veja as opções:",
    "Para escolher, aqui vão as opções:",
    "Para destravar, estas são as opções:",
    "Para continuar, estas são as opções:",
    "Antes de seguir, veja as opções:",
    "Para você escolher, estas são as opções:",
    "Vou te dar as opções:",
    "Vou apresentar as opções:",
]


CLARIFY_BANK_V119: List[str] = [
    "Não ficou claro. Você pode esclarecer o que você quer dizer?",
    "Ainda não entendi. Você pode esclarecer o que você quer dizer?",
    "Preciso de um detalhe. Você pode esclarecer o que você quer dizer?",
    "Só para eu não assumir errado: o que você quer dizer exatamente?",
    "Me ajude com um detalhe: o que você quer dizer com isso?",
    "Pode explicar de outro jeito? O que você quer dizer exatamente?",
    "Eu não consegui interpretar com segurança. Você pode esclarecer o que você quer dizer?",
    "Antes de seguir, preciso entender: o que você quer dizer com isso?",
    "Para eu responder certo, preciso de clarificação: o que você quer dizer exatamente?",
    "Fiquei em dúvida. Você pode esclarecer o que você quer dizer?",
    "Você pode ser mais específico sobre o que você quer dizer?",
    "Qual parte exatamente você quer que eu faça agora?",
    "O que exatamente você quer dizer por “isso” aqui?",
    "Você está se referindo a quê, exatamente?",
    "Quando você diz “isso”, qual coisa você quer dizer?",
    "Você quer que eu continue o plano ou quer mudar alguma coisa?",
    "Qual é o próximo passo que você quer: continuar, mudar, ou encerrar?",
    "Me diga só o alvo: o que você quer obter como resultado?",
    "Só preciso de uma informação: o que você quer dizer com isso?",
    "Preciso entender a referência: isso aponta para qual item/ação?",
    "O que você quer que eu considere como “isso” neste contexto?",
    "Você pode dizer explicitamente qual parte devo usar como referência?",
    "Você quer que eu responda, que eu pergunte um detalhe, ou que eu resuma?",
    "Você está pedindo uma ação específica ou uma explicação?",
    "Qual variável/assunto você quer usar agora?",
    "Qual é o dado que falta para eu continuar?",
    "Você quer que eu avance ou que eu confirme entendimento?",
    "Eu consigo seguir, mas preciso saber: o que exatamente você quer dizer?",
    "Para evitar assumir errado, esclareça: você quer dizer qual item?",
    "Você está falando do objetivo atual ou de outra coisa?",
    "Você está se referindo ao último passo, ao plano, ou ao objetivo?",
    "Você quer que eu continue do ponto onde parei ou que eu replaine?",
    "Qual parte do que eu disse você quer ajustar?",
    "Você pode apontar o que mudar (qual campo/variável/parte)?",
    "Qual é a opção correta aqui (A/B/C), ou você quer outra opção?",
    "Você quer que eu dê um exemplo ou que eu execute a ação?",
    "Qual é o item certo para eu usar como referência agora?",
    "Eu preciso de um nome/identificador: a que você está se referindo?",
    "Você está confirmando, recusando, ou pedindo para seguir?",
    "Você quer que eu siga exatamente igual ou que eu simplifique?",
    "Você quer um resumo curto ou um próximo passo concreto?",
    "Você quer que eu continue com o mesmo objetivo ou mudou o objetivo?",
    "Qual é a mudança que você quer: valor, prioridade, ou escopo?",
    "Você quer que eu escolha uma opção ou quer escolher você?",
    "Você pode dizer qual chave/variável/parte devo usar?",
    "Eu preciso de uma referência explícita. Qual é ela?",
    "O que você quer que eu faça com isso: explicar, executar, ou perguntar?",
    "Você pode descrever em uma frase o que você quer agora?",
    "Qual é a pergunta que você quer responder agora?",
    "Qual é o resultado final que você quer (em uma frase)?",
    "Qual é a restrição mais importante aqui?",
    "Qual é o detalhe que você quer definir primeiro?",
    "Você quer continuar do jeito atual ou quer revisar o plano?",
    "Você quer que eu dê opções A/B ou que eu faça uma pergunta específica?",
    "Você está se referindo ao que eu acabei de dizer ou a algo anterior?",
    "Você quer que eu continue a partir do último resultado ou do objetivo?",
    "Você está se referindo ao último valor, ao plano, ou ao objetivo ativo?",
    "Qual destes é “isso”: o objetivo, o plano, ou o último resultado?",
]


REFORMULATE_BANK_V119: List[str] = [
    "Não entendi. Você pode reformular usando um comando suportado (set/get/add/summary/end) ou descrever o objetivo?",
    "Não ficou claro. Você pode reformular usando um comando suportado (set/get/add/summary/end) ou descrever o objetivo?",
    "Preciso de clarificação. Reformule com set/get/add/summary/end, ou descreva o objetivo.",
    "Para eu continuar: use set/get/add/summary/end, ou descreva o objetivo em uma frase.",
    "Eu não consigo executar isso como comando. Use set/get/add/summary/end, ou descreva o objetivo.",
]


INVALID_COMMAND_BANK_V119: List[str] = [
    "Comando inválido: {msg}",
    "Entrada inválida: {msg}",
    "Pedido inválido: {msg}",
]


@dataclass(frozen=True)
class VariantChoiceV119:
    variant_kind: str
    variant_index: int
    text: str
    prefix2: str
    choice_sig: str

    def to_dict(self) -> Dict[str, Any]:
        body = {
            "schema_version": 119,
            "kind": "variant_choice_v119",
            "variant_kind": str(self.variant_kind),
            "variant_index": int(self.variant_index),
            "text": str(self.text),
            "text_sha256": _stable_hash_text(self.text),
            "prefix2": str(self.prefix2),
        }
        body["choice_sig"] = str(self.choice_sig)
        return dict(body)


def _choose_from_bank(
    *,
    variant_kind: str,
    bank: Sequence[str],
    context_sig: str,
    last_prefix2: str,
    attempt_index: int,
) -> VariantChoiceV119:
    b = [str(x) for x in bank if str(x)]
    if not b:
        txt = ""
        pref = ""
        cs = _stable_hash_obj({"k": str(variant_kind), "empty": True, "context_sig": str(context_sig), "attempt": int(attempt_index)})
        return VariantChoiceV119(variant_kind=str(variant_kind), variant_index=0, text=txt, prefix2=pref, choice_sig=str(cs))

    seed_obj = {"variant_kind": str(variant_kind), "context_sig": str(context_sig), "attempt_index": int(attempt_index)}
    seed_sig = _stable_hash_obj(seed_obj)
    idx0 = int(seed_sig, 16) % int(len(b))

    lp2 = str(last_prefix2 or "")
    lp2_toks = [t for t in lp2.split() if t]
    chosen_idx = int(idx0)
    chosen_text = str(b[chosen_idx])
    chosen_p2 = prefix2_from_text_v119(chosen_text)
    chosen_toks = _tokenize(chosen_text)
    chosen_first = chosen_toks[0] if chosen_toks else ""

    # Deterministic anti-collision: avoid repeating the same prefix2 when possible.
    #
    # Extra rule (bullet-aware):
    # If the previous prefix2 begins with a bullet token ("-" / "•"), the *effective*
    # prefix2 often becomes "<bullet> <first_word>". To reduce consecutive runs like
    # "- não" even when the raw template text doesn't include the bullet marker,
    # treat a repeated "<bullet> <first_word>" as a collision too.
    bullet_collision = False
    if lp2_toks and len(lp2_toks) >= 2 and lp2_toks[0] in {"-", "•"} and chosen_first:
        bullet_collision = str(chosen_first) == str(lp2_toks[1])

    if lp2 and chosen_p2 and (chosen_p2 == lp2 or bullet_collision) and len(b) > 1:
        for off in range(1, len(b) + 1):
            idx = (idx0 + off) % len(b)
            t = str(b[idx])
            p2 = prefix2_from_text_v119(t)
            toks2 = _tokenize(t)
            first2 = toks2[0] if toks2 else ""
            bullet_collision2 = False
            if lp2_toks and len(lp2_toks) >= 2 and lp2_toks[0] in {"-", "•"} and first2:
                bullet_collision2 = str(first2) == str(lp2_toks[1])
            if p2 and p2 != lp2 and not bullet_collision2:
                chosen_idx = int(idx)
                chosen_text = str(t)
                chosen_p2 = str(p2)
                chosen_first = str(first2)
                break

    choice_sig = _stable_hash_obj(
        {
            "schema_version": 119,
            "variant_kind": str(variant_kind),
            "context_sig": str(context_sig),
            "attempt_index": int(attempt_index),
            "idx0": int(idx0),
            "chosen_index": int(chosen_idx),
            "chosen_prefix2": str(chosen_p2),
            "chosen_first_token": str(chosen_first),
            "last_prefix2": str(lp2),
            "last_prefix2_tokens": list(lp2_toks),
        }
    )
    return VariantChoiceV119(
        variant_kind=str(variant_kind),
        variant_index=int(chosen_idx),
        text=str(chosen_text),
        prefix2=str(chosen_p2),
        choice_sig=str(choice_sig),
    )


def choose_ack_variant_v119(
    *,
    context_sig: str,
    last_prefix2: str,
    attempt_index: int,
    ) -> VariantChoiceV119:
    return _choose_from_bank(
        variant_kind="ack",
        bank=list(ACK_BANK_V119),
        context_sig=str(context_sig),
        last_prefix2=str(last_prefix2),
        attempt_index=int(attempt_index),
    )


def choose_options_preamble_variant_v119(
    *,
    context_sig: str,
    last_prefix2: str,
    attempt_index: int,
) -> VariantChoiceV119:
    return _choose_from_bank(
        variant_kind="options_preamble",
        bank=list(OPTIONS_PREAMBLE_BANK_V119),
        context_sig=str(context_sig),
        last_prefix2=str(last_prefix2),
        attempt_index=int(attempt_index),
    )


def choose_clarify_variant_v119(
    *,
    context_sig: str,
    last_prefix2: str,
    attempt_index: int,
) -> VariantChoiceV119:
    return _choose_from_bank(
        variant_kind="clarify",
        bank=list(CLARIFY_BANK_V119),
        context_sig=str(context_sig),
        last_prefix2=str(last_prefix2),
        attempt_index=int(attempt_index),
    )


def choose_reformulate_variant_v119(
    *,
    context_sig: str,
    last_prefix2: str,
    attempt_index: int,
) -> VariantChoiceV119:
    return _choose_from_bank(
        variant_kind="reformulate",
        bank=list(REFORMULATE_BANK_V119),
        context_sig=str(context_sig),
        last_prefix2=str(last_prefix2),
        attempt_index=int(attempt_index),
    )


def render_invalid_command_text_v119(
    *,
    msg: str,
    context_sig: str,
    last_prefix2: str,
    attempt_index: int,
) -> Tuple[str, VariantChoiceV119]:
    choice = _choose_from_bank(
        variant_kind="invalid_command",
        bank=[str(t).format(msg=str(msg)) for t in INVALID_COMMAND_BANK_V119],
        context_sig=str(context_sig),
        last_prefix2=str(last_prefix2),
        attempt_index=int(attempt_index),
    )
    return str(choice.text), choice
