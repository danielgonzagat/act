# V88 — `objective_csv_v88` (Objetivos como CSV/CSG)

## O que é
`objective_csv_v88` é um ACT persistente que representa um **critério de sucesso executável** no mesmo runtime determinístico usado para `concept_csv`. Em vez de depender apenas de `expected + validator`, um goal pode apontar para um objective que calcula um **veredito determinístico**.

## Inputs reservados
Quando usado para validar um goal, o executor injeta:

- `__output`: output bruto produzido pela execução do goal (string/dict; se não existir, `""`)
- `__goal`: resumo determinístico do goal (id, kind, bindings, expected, etc.)
- `__step`: step atual (int)

Além disso, é comum passar `expected` a partir do próprio goal (conveniência), mas isso é opcional: o objective define seu próprio contrato via `input_schema`.

## Formato do retorno (ObjectiveVerdictV88)
O runtime normaliza o output do objective para o contrato:

```json
{
  "ok": true,
  "score": 1,
  "reason": "",
  "details": {}
}
```

Regras de normalização:
- Se o programa retorna **bool**: `ok=bool`, `score=1/0`, `reason=""`, `details={}`
- Se retorna **dict**: extrai `ok/score/reason/details` com defaults determinísticos
- Qualquer outro tipo: falha (`ok=false`) com `reason` determinístico

## Como criar um objective simples (igualdade textual)
O exemplo incluído em `atos_core/objective_v88.py` cria um objective que compara:
- `__output` (str)
- `expected` (str)

Ele retorna um dict no formato `ObjectiveVerdictV88`.

Veja:
- `atos_core/objective_v88.py` → `make_objective_eq_text_act_v88(...)`

## Migração futura (V89+)
O caminho de migração natural é:
- transformar validators antigos em `objective_csv_v88` equivalentes,
- e gradualmente fazer goals apontarem para `objective_act_id` em vez de depender de `expected+validator`.

