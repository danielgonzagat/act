"""
meta_ka_v156.py - Meta-Knowledge Ascent: KA operando sobre o próprio KA

Este módulo implementa o potencializador mais poderoso da arquitetura:
KA criar conceitos sobre COMO criar conceitos.

O SALTO:
- Hoje: KA cria conceitos para resolver tarefas
- Meta: KA cria conceitos sobre como criar conceitos
  - Regras de mineração viram objetos
  - Estratégias de abstração competem entre si
  - O sistema aprende A APRENDER

Exemplos de meta-conceitos:
- "quando falhar por simetria → tente conceito de grupo"
- "quando depth estagnar → forçar composição cruzada"
- "quando reuse cair → matar conceitos rasos"

Schema version: 156
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .act import Act, canonical_json_dumps, sha256_hex

META_KA_SCHEMA_VERSION_V156 = 156


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Concept Types
# ─────────────────────────────────────────────────────────────────────────────


class MetaConceptType(Enum):
    """Tipos de meta-conceitos."""
    
    MINING_STRATEGY = auto()      # Como minerar conceitos
    COMPOSITION_RULE = auto()     # Como compor conceitos
    DEATH_CRITERION = auto()      # Quando matar conceitos
    GENERALIZATION_TRIGGER = auto()  # Quando generalizar
    FAILURE_RESPONSE = auto()     # Como reagir a falhas
    PRESSURE_ADJUSTMENT = auto()  # Como ajustar pressões


@dataclass
class FailurePattern:
    """Padrão de falha identificado."""
    
    pattern_id: str
    failure_type: str  # "symmetry", "composition", "depth", "search", "budget"
    frequency: int
    examples: List[str]  # Task IDs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "failure_type": self.failure_type,
            "frequency": self.frequency,
            "examples_count": len(self.examples),
        }


@dataclass
class MetaConcept:
    """
    Um conceito sobre conceitos.
    
    Meta-conceitos não resolvem tarefas diretamente.
    Eles governam COMO o KA cria e gerencia conceitos.
    """
    
    meta_id: str
    meta_type: MetaConceptType
    
    # Condição de ativação
    trigger_condition: Dict[str, Any]
    
    # Ação a tomar
    action: Dict[str, Any]
    
    # Métricas de eficácia
    times_triggered: int = 0
    times_successful: int = 0
    times_failed: int = 0
    
    # Custo e sobrevivência
    cost_bits: float = 0.0
    is_alive: bool = True
    creation_time: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        total = self.times_successful + self.times_failed
        return self.times_successful / max(1, total)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta_id": self.meta_id,
            "meta_type": self.meta_type.name,
            "trigger_condition": self.trigger_condition,
            "action": self.action,
            "times_triggered": self.times_triggered,
            "success_rate": self.success_rate,
            "cost_bits": self.cost_bits,
            "is_alive": self.is_alive,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Meta-KA: Knowledge Ascent sobre o próprio Knowledge Ascent
# ─────────────────────────────────────────────────────────────────────────────


class MetaKnowledgeAscent:
    """
    META-KA: O sistema que aprende A APRENDER.
    
    Observa:
    - Padrões de falha
    - Padrões de sucesso
    - Dinâmica de conceitos (criação, morte, composição)
    
    Cria:
    - Meta-conceitos que governam mineração
    - Regras que ajustam pressões
    - Estratégias que competem entre si
    """
    
    def __init__(self, *, mdl_pressure: float = 1.0) -> None:
        # Meta-conceitos ativos
        self._meta_concepts: Dict[str, MetaConcept] = {}
        
        # Histórico de falhas (para minerar padrões)
        self._failure_history: List[Dict[str, Any]] = []
        
        # Histórico de conceitos (para aprender dinâmica)
        self._concept_lifecycle: List[Dict[str, Any]] = []
        
        # Pressões ajustáveis
        self._mdl_pressure = mdl_pressure
        self._composition_pressure = 1.0
        self._death_pressure = 1.0
        
        # Métricas
        self._meta_concepts_created = 0
        self._meta_concepts_killed = 0
        self._pressure_adjustments = 0
        
        # Inicializar meta-conceitos base
        self._initialize_base_meta_concepts()
    
    def _initialize_base_meta_concepts(self) -> None:
        """Inicializa meta-conceitos fundamentais."""
        
        # Meta-conceito 1: Quando depth estagna, forçar composição
        self._create_meta_concept(
            meta_id="meta_depth_stagnation",
            meta_type=MetaConceptType.COMPOSITION_RULE,
            trigger_condition={
                "metric": "max_depth",
                "condition": "stagnant_for",
                "threshold": 10,  # 10 tarefas sem aumento de depth
            },
            action={
                "type": "force_composition",
                "min_depth": 2,
                "description": "Forçar composição de conceitos existentes",
            },
        )
        
        # Meta-conceito 2: Quando reuse cai, matar conceitos rasos
        self._create_meta_concept(
            meta_id="meta_low_reuse_death",
            meta_type=MetaConceptType.DEATH_CRITERION,
            trigger_condition={
                "metric": "avg_reuse",
                "condition": "below",
                "threshold": 1.5,
            },
            action={
                "type": "kill_shallow_concepts",
                "max_depth_to_kill": 1,
                "min_reuse_to_survive": 2,
                "description": "Matar conceitos rasos com baixo reuso",
            },
        )
        
        # Meta-conceito 3: Quando falha por simetria, tentar grupo
        self._create_meta_concept(
            meta_id="meta_symmetry_failure",
            meta_type=MetaConceptType.FAILURE_RESPONSE,
            trigger_condition={
                "failure_type": "symmetry_not_detected",
                "frequency": 3,  # 3 falhas do mesmo tipo
            },
            action={
                "type": "mine_symmetry_concept",
                "priority_ops": ["symmetry_fill_h", "symmetry_fill_v", "reflect_h", "reflect_v"],
                "description": "Minerar conceito de simetria",
            },
        )
        
        # Meta-conceito 4: Quando busca explode, aumentar pressão MDL
        self._create_meta_concept(
            meta_id="meta_search_explosion",
            meta_type=MetaConceptType.PRESSURE_ADJUSTMENT,
            trigger_condition={
                "metric": "search_budget_exceeded_rate",
                "condition": "above",
                "threshold": 0.5,  # Mais de 50% das tarefas excedem budget
            },
            action={
                "type": "increase_mdl_pressure",
                "factor": 1.2,
                "description": "Aumentar pressão MDL para forçar conceitos mais simples",
            },
        )
        
        # Meta-conceito 5: Quando transferência falha, forçar abstração
        self._create_meta_concept(
            meta_id="meta_transfer_failure",
            meta_type=MetaConceptType.GENERALIZATION_TRIGGER,
            trigger_condition={
                "metric": "transfer_success_rate",
                "condition": "below",
                "threshold": 0.3,
            },
            action={
                "type": "force_abstraction",
                "abstraction_level": "pattern_to_rule",
                "description": "Forçar abstração de padrões para regras",
            },
        )
    
    def _create_meta_concept(
        self,
        *,
        meta_id: str,
        meta_type: MetaConceptType,
        trigger_condition: Dict[str, Any],
        action: Dict[str, Any],
    ) -> MetaConcept:
        """Cria um meta-conceito."""
        
        mc = MetaConcept(
            meta_id=meta_id,
            meta_type=meta_type,
            trigger_condition=trigger_condition,
            action=action,
            cost_bits=self._compute_meta_cost(trigger_condition, action),
        )
        
        self._meta_concepts[meta_id] = mc
        self._meta_concepts_created += 1
        
        return mc
    
    def _compute_meta_cost(
        self,
        trigger: Dict[str, Any],
        action: Dict[str, Any],
    ) -> float:
        """Computa custo MDL de um meta-conceito."""
        # Custo proporcional à complexidade da condição e ação
        trigger_cost = len(json.dumps(trigger))
        action_cost = len(json.dumps(action))
        return (trigger_cost + action_cost) * 0.01
    
    # ─────────────────────────────────────────────────────────────────────────
    # Observação e Aprendizado
    # ─────────────────────────────────────────────────────────────────────────
    
    def observe_failure(
        self,
        task_id: str,
        failure_type: str,
        context: Dict[str, Any],
    ) -> None:
        """Observa uma falha para aprender padrões."""
        
        self._failure_history.append({
            "task_id": task_id,
            "failure_type": failure_type,
            "context": context,
            "timestamp": time.time(),
        })
        
        # Manter histórico limitado
        if len(self._failure_history) > 1000:
            self._failure_history = self._failure_history[-500:]
    
    def observe_concept_lifecycle(
        self,
        concept_id: str,
        event: str,  # "created", "used", "killed"
        context: Dict[str, Any],
    ) -> None:
        """Observa ciclo de vida de conceitos."""
        
        self._concept_lifecycle.append({
            "concept_id": concept_id,
            "event": event,
            "context": context,
            "timestamp": time.time(),
        })
        
        # Manter histórico limitado
        if len(self._concept_lifecycle) > 1000:
            self._concept_lifecycle = self._concept_lifecycle[-500:]
    
    def analyze_failure_patterns(self) -> List[FailurePattern]:
        """Analisa histórico de falhas para encontrar padrões."""
        
        # Agrupar por tipo de falha
        by_type: Dict[str, List[str]] = {}
        for f in self._failure_history:
            ft = f["failure_type"]
            if ft not in by_type:
                by_type[ft] = []
            by_type[ft].append(f["task_id"])
        
        patterns = []
        for failure_type, task_ids in by_type.items():
            if len(task_ids) >= 3:  # Padrão = 3+ ocorrências
                patterns.append(FailurePattern(
                    pattern_id=f"pattern_{failure_type}_{len(patterns)}",
                    failure_type=failure_type,
                    frequency=len(task_ids),
                    examples=task_ids[-10:],  # Últimos 10 exemplos
                ))
        
        return patterns
    
    # ─────────────────────────────────────────────────────────────────────────
    # Avaliação e Ativação de Meta-Conceitos
    # ─────────────────────────────────────────────────────────────────────────
    
    def evaluate_triggers(
        self,
        current_metrics: Dict[str, Any],
    ) -> List[Tuple[MetaConcept, Dict[str, Any]]]:
        """
        Avalia quais meta-conceitos devem ser ativados.
        
        Retorna lista de (meta_conceito, contexto_de_ativação).
        """
        
        triggered = []
        
        for mc in self._meta_concepts.values():
            if not mc.is_alive:
                continue
            
            if self._check_trigger(mc.trigger_condition, current_metrics):
                triggered.append((mc, {"metrics": current_metrics}))
                mc.times_triggered += 1
        
        return triggered
    
    def _check_trigger(
        self,
        condition: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> bool:
        """Verifica se uma condição de trigger é satisfeita."""
        
        metric_name = condition.get("metric")
        if metric_name and metric_name in metrics:
            value = metrics[metric_name]
            cond = condition.get("condition")
            threshold = condition.get("threshold", 0)
            
            if cond == "below":
                return value < threshold
            elif cond == "above":
                return value > threshold
            elif cond == "stagnant_for":
                # Verificar se métrica não mudou por N iterações
                stagnant_count = metrics.get(f"{metric_name}_stagnant_count", 0)
                return stagnant_count >= threshold
        
        # Verificar padrão de falha
        failure_type = condition.get("failure_type")
        if failure_type:
            frequency = condition.get("frequency", 1)
            recent_failures = [
                f for f in self._failure_history[-50:]
                if f["failure_type"] == failure_type
            ]
            return len(recent_failures) >= frequency
        
        return False
    
    def execute_action(
        self,
        mc: MetaConcept,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Executa a ação de um meta-conceito.
        
        Retorna resultado da ação.
        """
        
        action = mc.action
        action_type = action.get("type")
        
        result = {
            "meta_id": mc.meta_id,
            "action_type": action_type,
            "executed": True,
        }
        
        if action_type == "force_composition":
            result["directive"] = {
                "require_min_depth": action.get("min_depth", 2),
                "composition_mode": "cross_concept",
            }
            
        elif action_type == "kill_shallow_concepts":
            result["directive"] = {
                "death_filter": {
                    "max_depth": action.get("max_depth_to_kill", 1),
                    "min_reuse_to_survive": action.get("min_reuse_to_survive", 2),
                },
            }
            
        elif action_type == "mine_symmetry_concept":
            result["directive"] = {
                "mining_priority": action.get("priority_ops", []),
                "target_pattern": "symmetry",
            }
            
        elif action_type == "increase_mdl_pressure":
            factor = action.get("factor", 1.1)
            self._mdl_pressure *= factor
            self._pressure_adjustments += 1
            result["new_mdl_pressure"] = self._mdl_pressure
            
        elif action_type == "force_abstraction":
            result["directive"] = {
                "abstraction_level": action.get("abstraction_level"),
                "force_generalization": True,
            }
        
        return result
    
    def report_action_outcome(
        self,
        mc: MetaConcept,
        success: bool,
    ) -> None:
        """Reporta resultado de uma ação para ajustar meta-conceito."""
        
        if success:
            mc.times_successful += 1
        else:
            mc.times_failed += 1
        
        # Matar meta-conceito se taxa de sucesso muito baixa
        if mc.times_triggered >= 10 and mc.success_rate < 0.2:
            mc.is_alive = False
            self._meta_concepts_killed += 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Mineração de Novos Meta-Conceitos
    # ─────────────────────────────────────────────────────────────────────────
    
    def mine_new_meta_concepts(
        self,
        metrics_history: List[Dict[str, Any]],
    ) -> List[MetaConcept]:
        """
        Minera novos meta-conceitos a partir de padrões observados.
        
        Este é o coração do Meta-KA: criar regras sobre regras.
        """
        
        new_meta_concepts = []
        
        # Analisar padrões de falha
        patterns = self.analyze_failure_patterns()
        
        for pattern in patterns:
            # Verificar se já existe meta-conceito para este padrão
            existing = any(
                mc.trigger_condition.get("failure_type") == pattern.failure_type
                for mc in self._meta_concepts.values()
                if mc.is_alive
            )
            
            if not existing and pattern.frequency >= 5:
                # Criar meta-conceito para este padrão de falha
                mc = self._create_meta_concept(
                    meta_id=f"mined_{pattern.pattern_id}_{int(time.time())}",
                    meta_type=MetaConceptType.FAILURE_RESPONSE,
                    trigger_condition={
                        "failure_type": pattern.failure_type,
                        "frequency": 3,
                    },
                    action={
                        "type": "investigate_failure_pattern",
                        "pattern": pattern.failure_type,
                        "description": f"Investigar falhas de {pattern.failure_type}",
                    },
                )
                new_meta_concepts.append(mc)
        
        # Analisar dinâmica de métricas
        if len(metrics_history) >= 10:
            # Detectar estagnação
            recent = metrics_history[-10:]
            for metric in ["max_depth", "avg_reuse", "concepts_alive"]:
                values = [m.get(metric, 0) for m in recent]
                if len(set(values)) == 1:  # Mesma valor por 10 iterações
                    # Criar meta-conceito de resposta a estagnação
                    existing = any(
                        mc.trigger_condition.get("metric") == metric and
                        mc.trigger_condition.get("condition") == "stagnant_for"
                        for mc in self._meta_concepts.values()
                        if mc.is_alive
                    )
                    
                    if not existing:
                        mc = self._create_meta_concept(
                            meta_id=f"mined_stagnation_{metric}_{int(time.time())}",
                            meta_type=MetaConceptType.PRESSURE_ADJUSTMENT,
                            trigger_condition={
                                "metric": metric,
                                "condition": "stagnant_for",
                                "threshold": 10,
                            },
                            action={
                                "type": "shake_up",
                                "target_metric": metric,
                                "description": f"Shake up para destravar {metric}",
                            },
                        )
                        new_meta_concepts.append(mc)
        
        return new_meta_concepts
    
    # ─────────────────────────────────────────────────────────────────────────
    # Métricas e Relatórios
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do Meta-KA."""
        
        alive = [mc for mc in self._meta_concepts.values() if mc.is_alive]
        
        return {
            "schema_version": META_KA_SCHEMA_VERSION_V156,
            "meta_concepts_total": len(self._meta_concepts),
            "meta_concepts_alive": len(alive),
            "meta_concepts_created": self._meta_concepts_created,
            "meta_concepts_killed": self._meta_concepts_killed,
            "pressure_adjustments": self._pressure_adjustments,
            "current_mdl_pressure": self._mdl_pressure,
            "current_composition_pressure": self._composition_pressure,
            "current_death_pressure": self._death_pressure,
            "failure_history_size": len(self._failure_history),
            "concept_lifecycle_size": len(self._concept_lifecycle),
            "meta_concepts": [mc.to_dict() for mc in alive],
        }
    
    def get_active_directives(
        self,
        metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Retorna diretivas ativas dos meta-conceitos.
        
        Usado pelo KA principal para ajustar comportamento.
        """
        
        triggered = self.evaluate_triggers(metrics)
        directives = []
        
        for mc, ctx in triggered:
            result = self.execute_action(mc, ctx)
            if "directive" in result:
                directives.append({
                    "source": mc.meta_id,
                    "directive": result["directive"],
                })
        
        return directives


# ─────────────────────────────────────────────────────────────────────────────
# Aggressive MDL Pressure System
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConceptCostAnalysis:
    """Análise de custo de um conceito."""
    
    concept_id: str
    raw_cost_bits: float
    compression_ratio: float
    reuse_multiplier: float
    depth_bonus: float
    effective_cost: float
    
    survival_score: float  # > 1.0 = sobrevive, < 1.0 = morre
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "raw_cost_bits": self.raw_cost_bits,
            "compression_ratio": self.compression_ratio,
            "effective_cost": self.effective_cost,
            "survival_score": self.survival_score,
        }


class AggressiveMDLPressure:
    """
    Sistema de pressão MDL agressiva.
    
    Não basta conceitos funcionarem.
    Eles precisam CUSTAR POUCO.
    
    Quando você força:
    - Custo explícito
    - Compressão estrutural
    - Penalidade por complexidade
    - Competição real entre conceitos
    
    Acontece algo crítico:
    Conceitos começam a explicar mais com menos.
    
    Isso é exatamente:
    - O que humanos fazem
    - O que ciência faz
    - O que matemática faz
    """
    
    def __init__(
        self,
        *,
        base_pressure: float = 1.0,
        reuse_bonus_per_use: float = 0.1,
        depth_bonus_per_level: float = 0.05,
        death_threshold: float = 0.5,
    ) -> None:
        self.base_pressure = base_pressure
        self.reuse_bonus_per_use = reuse_bonus_per_use
        self.depth_bonus_per_level = depth_bonus_per_level
        self.death_threshold = death_threshold
        
        # Histórico de pressão
        self._pressure_history: List[float] = []
        self._death_events: List[Dict[str, Any]] = []
    
    def compute_concept_cost(
        self,
        concept_id: str,
        definition_bits: float,
        reuse_count: int,
        depth: int,
        tasks_explained: int,
        tasks_attempted: int,
    ) -> ConceptCostAnalysis:
        """
        Computa custo efetivo de um conceito.
        
        Fórmula:
        effective_cost = raw_cost / (compression * reuse_bonus * depth_bonus)
        survival_score = tasks_explained / (effective_cost * tasks_attempted)
        """
        
        raw_cost = definition_bits * self.base_pressure
        
        # Compression ratio: quanto o conceito comprime
        compression = max(0.1, tasks_explained / max(1, tasks_attempted))
        
        # Reuse bonus: conceitos reutilizados custam menos
        reuse_mult = 1.0 + (reuse_count * self.reuse_bonus_per_use)
        
        # Depth bonus: conceitos profundos são mais valiosos
        depth_bonus = 1.0 + (depth * self.depth_bonus_per_level)
        
        # Custo efetivo
        effective_cost = raw_cost / (compression * reuse_mult * depth_bonus)
        
        # Survival score
        if tasks_attempted > 0:
            survival = tasks_explained / (effective_cost * tasks_attempted + 0.1)
        else:
            survival = 0.5  # Conceito novo, neutro
        
        return ConceptCostAnalysis(
            concept_id=concept_id,
            raw_cost_bits=raw_cost,
            compression_ratio=compression,
            reuse_multiplier=reuse_mult,
            depth_bonus=depth_bonus,
            effective_cost=effective_cost,
            survival_score=survival,
        )
    
    def identify_death_candidates(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[ConceptCostAnalysis]:
        """
        Identifica conceitos que devem morrer.
        
        Conceitos morrem se:
        - survival_score < death_threshold
        - depth <= 1 E reuse < 2
        - compression_ratio < 0.1
        """
        
        death_candidates = []
        
        for c in concepts:
            analysis = self.compute_concept_cost(
                concept_id=c["concept_id"],
                definition_bits=c.get("cost_bits", 10.0),
                reuse_count=c.get("reuse_count", 0),
                depth=c.get("depth", 0),
                tasks_explained=c.get("tasks_solved", 0),
                tasks_attempted=c.get("tasks_attempted", 1),
            )
            
            should_die = False
            
            # Regra 1: survival_score muito baixo
            if analysis.survival_score < self.death_threshold:
                should_die = True
            
            # Regra 2: raso e sem reuso
            depth = c.get("depth", 0)
            reuse = c.get("reuse_count", 0)
            if depth <= 1 and reuse < 2:
                should_die = True
            
            # Regra 3: compressão muito baixa
            if analysis.compression_ratio < 0.1:
                should_die = True
            
            if should_die:
                death_candidates.append(analysis)
        
        return death_candidates
    
    def execute_death_pressure(
        self,
        concepts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Executa pressão de morte.
        
        Retorna conceitos que devem ser mortos e razões.
        """
        
        candidates = self.identify_death_candidates(concepts)
        
        deaths = []
        for cand in candidates:
            deaths.append({
                "concept_id": cand.concept_id,
                "survival_score": cand.survival_score,
                "effective_cost": cand.effective_cost,
                "reason": self._death_reason(cand),
            })
            
            self._death_events.append({
                "concept_id": cand.concept_id,
                "timestamp": time.time(),
                "survival_score": cand.survival_score,
            })
        
        return {
            "deaths_proposed": len(deaths),
            "deaths": deaths,
            "current_pressure": self.base_pressure,
        }
    
    def _death_reason(self, analysis: ConceptCostAnalysis) -> str:
        """Determina razão de morte."""
        
        if analysis.survival_score < self.death_threshold:
            return f"low_survival_score:{analysis.survival_score:.3f}"
        if analysis.compression_ratio < 0.1:
            return f"low_compression:{analysis.compression_ratio:.3f}"
        return "mdl_pressure_exceeded"
    
    def adjust_pressure(
        self,
        metrics: Dict[str, Any],
    ) -> float:
        """
        Ajusta pressão baseado em métricas.
        
        Se muitos conceitos morrendo → reduzir pressão
        Se conceitos estagnando → aumentar pressão
        """
        
        # Taxa de morte recente
        recent_deaths = len([
            d for d in self._death_events
            if time.time() - d["timestamp"] < 3600  # Última hora
        ])
        
        # Ajustar
        if recent_deaths > 10:
            # Muitas mortes → reduzir pressão
            self.base_pressure *= 0.95
        elif recent_deaths < 2:
            # Poucas mortes → aumentar pressão
            self.base_pressure *= 1.05
        
        # Limites
        self.base_pressure = max(0.5, min(3.0, self.base_pressure))
        
        self._pressure_history.append(self.base_pressure)
        
        return self.base_pressure
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do sistema de pressão."""
        
        return {
            "schema_version": META_KA_SCHEMA_VERSION_V156,
            "current_pressure": self.base_pressure,
            "death_events_total": len(self._death_events),
            "pressure_history_len": len(self._pressure_history),
            "reuse_bonus_per_use": self.reuse_bonus_per_use,
            "depth_bonus_per_level": self.depth_bonus_per_level,
            "death_threshold": self.death_threshold,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy as Law (Depth Requirement)
# ─────────────────────────────────────────────────────────────────────────────


class HierarchyLaw:
    """
    DEPTH COMO LEI, NÃO MÉTRICA.
    
    Regras:
    - Conceitos de nível 1 NÃO podem resolver tarefas difíceis
    - Só conceitos compostos resolvem tarefas compostas
    - Conceitos rasos morrem automaticamente em tarefas difíceis
    
    Isso força:
    - Conceitos de conceitos
    - Composição real
    - Abstração verdadeira
    """
    
    def __init__(
        self,
        *,
        min_depth_for_hard_tasks: int = 2,
        task_difficulty_threshold: float = 0.7,
    ) -> None:
        self.min_depth_for_hard_tasks = min_depth_for_hard_tasks
        self.task_difficulty_threshold = task_difficulty_threshold
        
        self._violations: List[Dict[str, Any]] = []
    
    def check_depth_requirement(
        self,
        task_difficulty: float,
        solution_depth: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifica se profundidade da solução é adequada para a tarefa.
        
        Tarefas difíceis EXIGEM soluções profundas.
        """
        
        if task_difficulty >= self.task_difficulty_threshold:
            if solution_depth < self.min_depth_for_hard_tasks:
                reason = (
                    f"task_difficulty={task_difficulty:.2f} >= {self.task_difficulty_threshold}, "
                    f"but solution_depth={solution_depth} < {self.min_depth_for_hard_tasks}"
                )
                self._violations.append({
                    "task_difficulty": task_difficulty,
                    "solution_depth": solution_depth,
                    "reason": reason,
                    "timestamp": time.time(),
                })
                return False, reason
        
        return True, None
    
    def filter_shallow_solutions(
        self,
        solutions: List[Dict[str, Any]],
        task_difficulty: float,
    ) -> List[Dict[str, Any]]:
        """
        Filtra soluções que não atendem requisito de profundidade.
        
        Para tarefas difíceis, REJEITA soluções rasas.
        """
        
        if task_difficulty < self.task_difficulty_threshold:
            return solutions  # Tarefas fáceis aceitam qualquer solução
        
        filtered = []
        for sol in solutions:
            depth = sol.get("depth", 0)
            if depth >= self.min_depth_for_hard_tasks:
                filtered.append(sol)
        
        return filtered
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas da lei de hierarquia."""
        
        return {
            "schema_version": META_KA_SCHEMA_VERSION_V156,
            "min_depth_for_hard_tasks": self.min_depth_for_hard_tasks,
            "task_difficulty_threshold": self.task_difficulty_threshold,
            "violations_total": len(self._violations),
            "recent_violations": self._violations[-10:],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Unified Power Amplifier
# ─────────────────────────────────────────────────────────────────────────────


class KAPowerAmplifier:
    """
    Amplificador de poder do KA.
    
    Integra todos os potencializadores:
    1. Meta-KA (KA sobre KA)
    2. Pressão MDL agressiva
    3. Lei de hierarquia (depth como lei)
    4. Análise de falhas como recurso
    """
    
    def __init__(
        self,
        *,
        mdl_pressure: float = 1.0,
        min_depth_hard: int = 2,
        difficulty_threshold: float = 0.7,
    ) -> None:
        self.meta_ka = MetaKnowledgeAscent(mdl_pressure=mdl_pressure)
        self.mdl_pressure = AggressiveMDLPressure(base_pressure=mdl_pressure)
        self.hierarchy_law = HierarchyLaw(
            min_depth_for_hard_tasks=min_depth_hard,
            task_difficulty_threshold=difficulty_threshold,
        )
    
    def process_iteration(
        self,
        *,
        current_metrics: Dict[str, Any],
        concepts: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Processa uma iteração do amplificador.
        
        Retorna diretivas para o KA principal.
        """
        
        # 1. Observar falhas
        for f in failures:
            self.meta_ka.observe_failure(
                task_id=f.get("task_id", "unknown"),
                failure_type=f.get("failure_type", "unknown"),
                context=f,
            )
        
        # 2. Avaliar meta-conceitos
        meta_directives = self.meta_ka.get_active_directives(current_metrics)
        
        # 3. Avaliar pressão MDL
        death_result = self.mdl_pressure.execute_death_pressure(concepts)
        
        # 4. Ajustar pressão
        new_pressure = self.mdl_pressure.adjust_pressure(current_metrics)
        
        return {
            "schema_version": META_KA_SCHEMA_VERSION_V156,
            "meta_directives": meta_directives,
            "death_candidates": death_result["deaths"],
            "adjusted_pressure": new_pressure,
            "meta_ka_metrics": self.meta_ka.get_metrics(),
            "mdl_metrics": self.mdl_pressure.get_metrics(),
            "hierarchy_metrics": self.hierarchy_law.get_metrics(),
        }
    
    def get_full_metrics(self) -> Dict[str, Any]:
        """Retorna métricas completas do amplificador."""
        
        return {
            "schema_version": META_KA_SCHEMA_VERSION_V156,
            "meta_ka": self.meta_ka.get_metrics(),
            "mdl_pressure": self.mdl_pressure.get_metrics(),
            "hierarchy_law": self.hierarchy_law.get_metrics(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 70)
    print("META-KA V156 - Knowledge Ascent sobre Knowledge Ascent")
    print("=" * 70)
    
    # Criar amplificador
    amp = KAPowerAmplifier(
        mdl_pressure=1.0,
        min_depth_hard=2,
        difficulty_threshold=0.7,
    )
    
    # Simular algumas falhas
    print("\n📊 Simulando falhas...")
    for i in range(10):
        amp.meta_ka.observe_failure(
            task_id=f"task_{i}",
            failure_type="symmetry_not_detected" if i % 3 == 0 else "search_budget_exceeded",
            context={"iteration": i},
        )
    
    # Simular métricas
    metrics = {
        "max_depth": 2,
        "avg_reuse": 1.2,
        "concepts_alive": 15,
        "search_budget_exceeded_rate": 0.6,
    }
    
    # Simular conceitos
    concepts = [
        {"concept_id": "c1", "cost_bits": 10, "reuse_count": 1, "depth": 1, "tasks_solved": 2, "tasks_attempted": 10},
        {"concept_id": "c2", "cost_bits": 15, "reuse_count": 5, "depth": 2, "tasks_solved": 8, "tasks_attempted": 10},
        {"concept_id": "c3", "cost_bits": 20, "reuse_count": 0, "depth": 0, "tasks_solved": 1, "tasks_attempted": 10},
    ]
    
    print("\n📊 Processando iteração...")
    result = amp.process_iteration(
        current_metrics=metrics,
        concepts=concepts,
        failures=[],
    )
    
    print(f"\n🎯 Meta-diretivas ativas: {len(result['meta_directives'])}")
    for d in result['meta_directives']:
        print(f"   - {d['source']}: {d['directive']}")
    
    print(f"\n💀 Candidatos a morte: {len(result['death_candidates'])}")
    for d in result['death_candidates']:
        print(f"   - {d['concept_id']}: {d['reason']}")
    
    print(f"\n📈 Pressão ajustada: {result['adjusted_pressure']:.3f}")
    
    # Métricas finais
    full_metrics = amp.get_full_metrics()
    print(f"\n📊 Meta-conceitos ativos: {full_metrics['meta_ka']['meta_concepts_alive']}")
    print(f"📊 Meta-conceitos criados: {full_metrics['meta_ka']['meta_concepts_created']}")
    
    print("\n" + "=" * 70)
    print("Meta-KA V156 funcionando!")
    print("=" * 70)
