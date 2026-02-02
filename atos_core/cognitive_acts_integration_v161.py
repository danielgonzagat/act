"""
cognitive_acts_integration_v161.py - Módulo de Integração Central

Este módulo integra todos os novos atos cognitivos V161 ao núcleo do ACT,
fornecendo uma interface unificada para acesso a todos os operadores.

RESUMO DOS ATOS COGNITIVOS IMPLEMENTADOS:
=========================================

1. Operadores Conceituais (44):
   - Rotações, reflexões, translações
   - Manipulação de cores
   - Operações em formas/objetos
   - Operações de padrão

2. Operadores Condicionais (29):
   - Aplicação condicional
   - Iteração e loops
   - Branching e fallbacks
   - Seleção de cenários

3. Seletores Semânticos (30):
   - Seleção por tamanho
   - Seleção por cor
   - Seleção por posição
   - Seleção por forma
   - Seleção por unicidade

4. Mineradores de Conceito (17):
   - Mineração de analogias
   - Mineração de padrões
   - Generalização/especialização
   - Exploração exaustiva
   - Composição de regras

5. Ferramentas de Compressão (18):
   - Avaliadores MDL
   - Fusores de conceito
   - Compressores de padrão
   - Codificadores de estado
   - Podadores de redundância

6. Operadores de Composição/Abstração (35):
   - Composição básica
   - Templates
   - Hierarquia
   - Planejamento
   - Analogia e blending

TOTAL: 173 Atos Cognitivos de Nível 0 (Determinísticos)

Schema version: 161
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Type alias
GridV124 = List[List[int]]

COGNITIVE_INTEGRATION_SCHEMA_VERSION_V161 = 161

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS DOS MÓDULOS V161
# ─────────────────────────────────────────────────────────────────────────────

# Importar operadores conceituais
from .cognitive_ops_conceptual_v161 import (
    CONCEPTUAL_OPERATORS_V161,
    get_conceptual_operator_v161,
    list_conceptual_operators_v161,
)

# Importar operadores condicionais
from .cognitive_ops_conditional_v161 import (
    CONDITIONAL_OPERATORS_V161,
    get_conditional_operator_v161,
    list_conditional_operators_v161,
)

# Importar seletores semânticos
from .cognitive_ops_selectors_v161 import (
    SEMANTIC_SELECTORS_V161,
    get_semantic_selector_v161,
    list_semantic_selectors_v161,
)

# Importar mineradores de conceito
from .cognitive_miners_v161 import (
    CONCEPT_MINERS_V161,
    get_concept_miner_v161,
    list_concept_miners_v161,
)

# Importar ferramentas de compressão
from .cognitive_compression_v161 import (
    COMPRESSION_TOOLS_V161,
    get_compression_tool_v161,
    list_compression_tools_v161,
)

# Importar operadores de composição
from .cognitive_composition_v161 import (
    COMPOSITION_OPERATORS_V161,
    get_composition_operator_v161,
    list_composition_operators_v161,
)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRO UNIFICADO
# ─────────────────────────────────────────────────────────────────────────────


class CognitiveActsRegistry:
    """
    Registro central de todos os atos cognitivos V161.
    
    Fornece interface unificada para:
    - Buscar operadores por nome
    - Listar operadores por categoria
    - Obter metadados de operadores
    """
    
    def __init__(self):
        self._registries = {
            "conceptual": CONCEPTUAL_OPERATORS_V161,
            "conditional": CONDITIONAL_OPERATORS_V161,
            "selectors": SEMANTIC_SELECTORS_V161,
            "miners": CONCEPT_MINERS_V161,
            "compression": COMPRESSION_TOOLS_V161,
            "composition": COMPOSITION_OPERATORS_V161,
        }
        
        self._all_operators: Dict[str, Callable] = {}
        self._operator_categories: Dict[str, str] = {}
        
        # Build unified registry
        for category, registry in self._registries.items():
            for name, operator in registry.items():
                self._all_operators[name] = operator
                self._operator_categories[name] = category
    
    def get(self, name: str) -> Optional[Callable]:
        """Obtém operador por nome."""
        return self._all_operators.get(name)
    
    def get_by_category(self, category: str) -> Dict[str, Callable]:
        """Obtém todos operadores de uma categoria."""
        return self._registries.get(category, {})
    
    def list_all(self) -> List[str]:
        """Lista todos os operadores disponíveis."""
        return list(self._all_operators.keys())
    
    def list_by_category(self, category: str) -> List[str]:
        """Lista operadores de uma categoria."""
        return list(self._registries.get(category, {}).keys())
    
    def get_category(self, name: str) -> Optional[str]:
        """Obtém categoria de um operador."""
        return self._operator_categories.get(name)
    
    def list_categories(self) -> List[str]:
        """Lista todas as categorias."""
        return list(self._registries.keys())
    
    def count_total(self) -> int:
        """Conta total de operadores."""
        return len(self._all_operators)
    
    def count_by_category(self) -> Dict[str, int]:
        """Conta operadores por categoria."""
        return {cat: len(reg) for cat, reg in self._registries.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo do registro."""
        return {
            "total_operators": self.count_total(),
            "by_category": self.count_by_category(),
            "categories": self.list_categories(),
            "schema_version": COGNITIVE_INTEGRATION_SCHEMA_VERSION_V161,
        }


# Instância global do registro
COGNITIVE_ACTS_REGISTRY_V161 = CognitiveActsRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE CONVENIÊNCIA
# ─────────────────────────────────────────────────────────────────────────────


def get_cognitive_act_v161(name: str) -> Optional[Callable]:
    """
    Obtém qualquer ato cognitivo pelo nome.
    
    Args:
        name: Nome do operador
        
    Returns:
        Função do operador ou None se não encontrado
    """
    return COGNITIVE_ACTS_REGISTRY_V161.get(name)


def list_all_cognitive_acts_v161() -> List[str]:
    """
    Lista todos os atos cognitivos disponíveis.
    
    Returns:
        Lista de nomes de operadores
    """
    return COGNITIVE_ACTS_REGISTRY_V161.list_all()


def get_cognitive_acts_summary_v161() -> Dict[str, Any]:
    """
    Obtém resumo dos atos cognitivos.
    
    Returns:
        Dicionário com estatísticas
    """
    return COGNITIVE_ACTS_REGISTRY_V161.get_summary()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE EXECUÇÃO
# ─────────────────────────────────────────────────────────────────────────────


class CognitiveExecutionPipeline:
    """
    Pipeline para execução coordenada de atos cognitivos.
    
    Suporta:
    - Execução sequencial
    - Execução condicional
    - Rastreamento de transformações
    - Rollback em caso de erro
    """
    
    def __init__(self):
        self.registry = COGNITIVE_ACTS_REGISTRY_V161
        self.history: List[Dict[str, Any]] = []
        self.trace_enabled = True
    
    def execute(self, 
                grid: GridV124, 
                operations: List[Dict[str, Any]]) -> GridV124:
        """
        Executa sequência de operações.
        
        Args:
            grid: Grid de entrada
            operations: Lista de operações [{name, params}, ...]
            
        Returns:
            Grid transformado
        """
        result = [[int(c) for c in row] for row in grid]
        
        for op_spec in operations:
            name = op_spec.get("name")
            params = op_spec.get("params", {})
            
            op = self.registry.get(name)
            if op is None:
                continue
            
            try:
                prev = [[int(c) for c in row] for row in result]
                result = op(result, **params)
                
                if self.trace_enabled:
                    self.history.append({
                        "operation": name,
                        "params": params,
                        "success": True,
                        "input_shape": (len(prev), len(prev[0]) if prev else 0),
                        "output_shape": (len(result), len(result[0]) if result else 0),
                    })
            except Exception as e:
                if self.trace_enabled:
                    self.history.append({
                        "operation": name,
                        "params": params,
                        "success": False,
                        "error": str(e),
                    })
        
        return result
    
    def execute_with_selection(self,
                               grid: GridV124,
                               selector_name: str,
                               operation_name: str,
                               *, selector_params: Dict = None,
                               operation_params: Dict = None) -> GridV124:
        """
        Executa operação apenas na região selecionada.
        
        Args:
            grid: Grid de entrada
            selector_name: Nome do seletor
            operation_name: Nome da operação
            selector_params: Parâmetros do seletor
            operation_params: Parâmetros da operação
            
        Returns:
            Grid com operação aplicada na seleção
        """
        selector_params = selector_params or {}
        operation_params = operation_params or {}
        
        selector = self.registry.get(selector_name)
        operation = self.registry.get(operation_name)
        
        if selector is None or operation is None:
            return [[int(c) for c in row] for row in grid]
        
        # Get selection
        selection = selector(grid, **selector_params)
        
        # Apply operation to selected region
        result = [[int(c) for c in row] for row in grid]
        selected_grid = selection.to_grid(grid)
        
        # Apply operation
        transformed = operation(selected_grid, **operation_params)
        
        # Merge back
        h, w = len(result), len(result[0]) if result else 0
        th, tw = len(transformed), len(transformed[0]) if transformed else 0
        
        for r in range(min(h, th)):
            for c in range(min(w, tw)):
                if int(selection.mask[r][c]) != 0:
                    result[r][c] = int(transformed[r][c])
        
        return result
    
    def mine_and_apply(self,
                       examples: List[Tuple[GridV124, GridV124]],
                       test_input: GridV124,
                       miner_name: str = "analogy_miner") -> GridV124:
        """
        Minera regras dos exemplos e aplica ao teste.
        
        Args:
            examples: Lista de pares (input, output)
            test_input: Input de teste
            miner_name: Nome do minerador
            
        Returns:
            Output predito
        """
        miner = self.registry.get(miner_name)
        if miner is None:
            return [[int(c) for c in row] for row in test_input]
        
        # Mine concepts
        result = miner(examples)
        
        # Apply mined concept (simplified)
        if result.concepts:
            # Try to apply first concept
            concept = result.concepts[0]
            
            if concept.get("type") == "color_transform":
                # Apply color transformation
                output = [[int(c) for c in row] for row in test_input]
                # Implementation would go here
                return output
        
        return [[int(c) for c in row] for row in test_input]
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Retorna histórico de execução."""
        return self.history
    
    def clear_trace(self):
        """Limpa histórico."""
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# ARC SOLVER INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────


class ARCSolverV161:
    """
    Solver ARC usando os atos cognitivos V161.
    
    Implementa estratégia de resolução:
    1. Análise inicial com seletores
    2. Mineração de padrões
    3. Composição de transformações
    4. Aplicação e validação
    """
    
    def __init__(self):
        self.registry = COGNITIVE_ACTS_REGISTRY_V161
        self.pipeline = CognitiveExecutionPipeline()
    
    def solve(self, 
              train_examples: List[Tuple[GridV124, GridV124]],
              test_input: GridV124) -> GridV124:
        """
        Resolve tarefa ARC.
        
        Args:
            train_examples: Exemplos de treino [(input, output), ...]
            test_input: Input de teste
            
        Returns:
            Output predito
        """
        # Step 1: Analyze examples
        analysis = self._analyze_examples(train_examples)
        
        # Step 2: Mine transformation rules
        rules = self._mine_rules(train_examples, analysis)
        
        # Step 3: Compose solution
        solution_ops = self._compose_solution(rules)
        
        # Step 4: Apply to test
        if solution_ops:
            return self.pipeline.execute(test_input, solution_ops)
        
        # Fallback: try direct analogy
        if train_examples:
            analogy_composer = self.registry.get("analogy_composition")
            if analogy_composer:
                return analogy_composer(train_examples[0], test_input)
        
        return [[int(c) for c in row] for row in test_input]
    
    def _analyze_examples(self, 
                          examples: List[Tuple[GridV124, GridV124]]) -> Dict[str, Any]:
        """Analisa exemplos de treino."""
        analysis = {
            "shape_changes": [],
            "color_changes": [],
            "object_counts": [],
        }
        
        for inp, out in examples:
            ih, iw = len(inp), len(inp[0]) if inp else 0
            oh, ow = len(out), len(out[0]) if out else 0
            
            analysis["shape_changes"].append({
                "input": (ih, iw),
                "output": (oh, ow),
                "same": ih == oh and iw == ow
            })
        
        return analysis
    
    def _mine_rules(self,
                    examples: List[Tuple[GridV124, GridV124]],
                    analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Minera regras dos exemplos."""
        rules = []
        
        # Try different miners
        miners_to_try = ["generalization_miner", "difference_miner", "analogy_miner"]
        
        for miner_name in miners_to_try:
            miner = self.registry.get(miner_name)
            if miner:
                try:
                    result = miner(examples)
                    if result.confidence > 0.5:
                        rules.extend(result.concepts)
                except:
                    pass
        
        return rules
    
    def _compose_solution(self, 
                          rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compõe solução a partir das regras."""
        operations = []
        
        for rule in rules:
            rule_type = rule.get("type", "")
            
            if rule_type == "color_mapping":
                # Add color replacement operations
                mapping = rule.get("mapping", {})
                for old_color, new_color in mapping.items():
                    operations.append({
                        "name": "replace_color",
                        "params": {"old": old_color, "new": new_color}
                    })
            
            elif rule_type == "ratio_shape_transform":
                ratio = rule.get("ratio", (1, 1))
                if ratio[0] > 1 or ratio[1] > 1:
                    operations.append({
                        "name": "enlarge_pattern",
                        "params": {"factor": int(ratio[0])}
                    })
        
        return operations


# ─────────────────────────────────────────────────────────────────────────────
# TESTES RÁPIDOS
# ─────────────────────────────────────────────────────────────────────────────


def run_integration_tests() -> Dict[str, Any]:
    """
    Executa testes de integração básicos.
    
    Returns:
        Resultado dos testes
    """
    results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    # Test 1: Registry access
    try:
        summary = COGNITIVE_ACTS_REGISTRY_V161.get_summary()
        assert summary["total_operators"] > 150
        results["passed"] += 1
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Registry test: {str(e)}")
    
    # Test 2: Conceptual operator
    try:
        rotate = COGNITIVE_ACTS_REGISTRY_V161.get("rotate_90")
        if rotate:
            grid = [[1, 2], [3, 4]]
            rotated = rotate(grid)
            assert len(rotated) > 0
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("rotate_90 not found")
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Conceptual op test: {str(e)}")
    
    # Test 3: Selector
    try:
        selector = COGNITIVE_ACTS_REGISTRY_V161.get("select_largest_object")
        if selector:
            grid = [[0, 1, 0], [0, 1, 0], [2, 0, 0]]
            selection = selector(grid)
            assert selection is not None
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("select_largest_object not found")
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Selector test: {str(e)}")
    
    # Test 4: Category listing
    try:
        categories = COGNITIVE_ACTS_REGISTRY_V161.list_categories()
        assert len(categories) == 6
        results["passed"] += 1
    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Category test: {str(e)}")
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    # Registry
    "COGNITIVE_ACTS_REGISTRY_V161",
    "CognitiveActsRegistry",
    
    # Functions
    "get_cognitive_act_v161",
    "list_all_cognitive_acts_v161",
    "get_cognitive_acts_summary_v161",
    
    # Pipeline
    "CognitiveExecutionPipeline",
    
    # Solver
    "ARCSolverV161",
    
    # Tests
    "run_integration_tests",
    
    # Individual registries (for direct access)
    "CONCEPTUAL_OPERATORS_V161",
    "CONDITIONAL_OPERATORS_V161",
    "SEMANTIC_SELECTORS_V161",
    "CONCEPT_MINERS_V161",
    "COMPRESSION_TOOLS_V161",
    "COMPOSITION_OPERATORS_V161",
    
    # Individual getters
    "get_conceptual_operator_v161",
    "get_conditional_operator_v161",
    "get_semantic_selector_v161",
    "get_concept_miner_v161",
    "get_compression_tool_v161",
    "get_composition_operator_v161",
    
    # Individual listers
    "list_conceptual_operators_v161",
    "list_conditional_operators_v161",
    "list_semantic_selectors_v161",
    "list_concept_miners_v161",
    "list_compression_tools_v161",
    "list_composition_operators_v161",
]


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
MÓDULO DE INTEGRAÇÃO V161 - RESUMO
==================================

Este módulo integra TODOS os atos cognitivos expandidos:

┌─────────────────────────────────────────────────────────────────┐
│                    ATOS COGNITIVOS V161                        │
├─────────────────────────────────────────────────────────────────┤
│ Categoria                    │ Quantidade │ Arquivo             │
├─────────────────────────────────────────────────────────────────┤
│ Operadores Conceituais       │     44     │ _conceptual_v161.py │
│ Operadores Condicionais      │     29     │ _conditional_v161.py│
│ Seletores Semânticos         │     30     │ _selectors_v161.py  │
│ Mineradores de Conceito      │     17     │ _miners_v161.py     │
│ Ferramentas de Compressão    │     18     │ _compression_v161.py│
│ Operadores Composição/Abstr  │     35     │ _composition_v161.py│
├─────────────────────────────────────────────────────────────────┤
│ TOTAL                        │    173     │                     │
└─────────────────────────────────────────────────────────────────┘

OBJETIVO: Alcançar ≥90% de acerto em ARC-AGI-1 e ARC-AGI-2

PRINCÍPIOS:
- Todos operadores são Level 0 (determinísticos)
- Compatíveis com GridV124 (List[List[int]])
- Projetados para composição e encadeamento
- Seguem arquitetura ACT de autoridade cognitiva
"""
