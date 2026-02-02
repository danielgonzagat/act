# STATUS_PACK_V158_SHARED_CACHE_ANALYSIS.md

## Objetivo Original
Implementar cache compartilhado global entre workers no solver paralelo ARC.

## Experimento Realizado

### Hipótese
Cache compartilhado via `multiprocessing.Manager` permitiria reutilizar resultados entre workers, acelerando a busca.

### Implementação
Criado `arc_parallel_solver_v158.py` com:
- `SharedCacheManager` usando `Manager().dict()` para IPC
- Caches compartilhados: `apply_cache`, `eval_cache`, `grid_hash_cache`
- Workers acessando caches via IPC

### Benchmark (10 tarefas ARC, depth=3, programs=500, workers=4)

| Configuração | Tempo Total | Cache Hits |
|--------------|-------------|------------|
| **Com Manager** | 47.84s | apply=84,137, eval=46,473 |
| **Sem Manager** | 24.18s | N/A (local apenas) |

**Speedup com Manager: 0.51x (MAIS LENTO!)**

## Análise da Causa

1. **IPC Overhead Domina**: `multiprocessing.Manager` usa sockets/pipes
2. **Operações Pequenas**: Cada op de grid é O(n²) com n pequeno
3. **Latência > Compute**: Tempo de IPC > tempo de recomputar localmente

## Conclusão Técnica

Para busca combinatória em ARC:
- **LOCAL caches são MAIS RÁPIDOS** que shared via IPC
- **Paralelismo real** vem de PARTICIONAR o espaço de busca
- Cada worker explora operadores disjuntos no primeiro nível
- Caches locais já capturam redundância intra-branch

## Código Final

`arc_parallel_solver_v158.py` mantém a API de "shared cache" mas internamente usa apenas caches locais, pois é mais rápido.

```python
# Configuração (enable_shared_cache é aceito mas ignorado internamente)
config = ParallelSolverConfigV158(
    num_workers=8,
    max_programs_per_worker=500,
    max_depth=4,
    enable_shared_cache=True,  # Aceito para compatibilidade, mas local é usado
)
```

## Testes
- 9 testes específicos para V158
- 517 testes totais passando

## Próximos Passos para Performance

1. **Shared Memory (mmap)**: Para dados read-only como paletas e shapes
2. **Work Stealing**: Workers que terminam cedo ajudam os lentos
3. **Pre-compute Applicability**: Cache de quais ops aplicam a qual signature

## Lição Aprendida

> "Otimização prematura é a raiz de todo mal, mas medição prematura é a raiz de toda sabedoria." - Adaptado de Knuth

Sempre fazer benchmark ANTES de assumir que compartilhamento ajuda.
