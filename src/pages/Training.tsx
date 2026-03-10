import {
  Brain,
  Cpu,
  Moon,
  FlaskConical,
  Clock,
  Zap,
  ArrowRight,
} from 'lucide-react';
import type { TrainingExperiment } from '../types';

const infoCards = [
  {
    icon: Brain,
    title: 'Modelo Base',
    description: 'EfficientNet-B0 com transfer learning',
    detail:
      'Pré-treinado no ImageNet, fine-tuned para classificação de doenças foliares da soja com 7 classes.',
    color: 'bg-primary/15 text-primary',
  },
  {
    icon: Cpu,
    title: 'Hardware',
    description: 'Mac Mini M4, GPU MPS',
    detail:
      'Treinamento local usando Metal Performance Shaders para aceleração GPU no Apple Silicon.',
    color: 'bg-accent/15 text-accent',
  },
  {
    icon: Moon,
    title: 'Meta',
    description: '~100 experimentos/noite',
    detail:
      'Execução autônoma durante a noite, explorando hiperparâmetros automaticamente enquanto você dorme.',
    color: 'bg-success/15 text-success',
  },
  {
    icon: FlaskConical,
    title: 'Método',
    description: 'autoresearch adaptado do Karpathy',
    detail:
      'Inspirado no conceito de Andrej Karpathy: o agente gera hipóteses, treina, avalia e itera autonomamente.',
    color: 'bg-warning/15 text-warning',
  },
];

const emptyExperiments: TrainingExperiment[] = [];

const tableHeaders = ['Modelo', 'Acurácia', 'Config', 'Status', 'Data'];

export default function Training() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-3 flex-wrap">
        <h1 className="text-2xl font-bold text-text">
          Treinamento Autônomo
        </h1>
        <span className="px-3 py-1 rounded-full bg-accent/15 text-accent text-sm font-medium">
          Em breve
        </span>
      </div>

      {/* Status */}
      <div className="bg-surface rounded-xl p-6 flex items-center gap-4">
        <div className="p-3 rounded-full bg-warning/15">
          <Clock size={24} className="text-warning" />
        </div>
        <div>
          <p className="text-lg font-semibold text-text">
            Etapa 2 — Aguardando dataset
          </p>
          <p className="text-sm text-text-secondary mt-0.5">
            O treinamento será iniciado quando o dataset atingir volume mínimo
            por classe.
          </p>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {infoCards.map((card) => {
          const Icon = card.icon;
          return (
            <div key={card.title} className="bg-surface rounded-xl p-6">
              <div className={`inline-flex p-3 rounded-full ${card.color} mb-4`}>
                <Icon size={22} />
              </div>
              <h3 className="text-base font-semibold text-text">
                {card.title}
              </h3>
              <p className="text-sm text-primary-light font-medium mt-1">
                {card.description}
              </p>
              <p className="text-sm text-text-secondary mt-2 leading-relaxed">
                {card.detail}
              </p>
            </div>
          );
        })}
      </div>

      {/* Architecture Diagram */}
      <section>
        <h2 className="text-xl font-bold text-text mb-4">
          Arquitetura do Pipeline
        </h2>
        <div className="bg-surface rounded-xl p-6 overflow-x-auto">
          <div className="flex items-center gap-3 min-w-[600px] text-sm">
            <div className="flex flex-col items-center gap-1 px-4 py-3 bg-primary/10 rounded-lg border border-primary/20">
              <Zap size={16} className="text-primary" />
              <span className="text-text font-medium">Dataset</span>
              <span className="text-xs text-text-secondary">Imagens + Labels</span>
            </div>

            <ArrowRight size={18} className="text-text-secondary shrink-0" />

            <div className="flex flex-col items-center gap-1 px-4 py-3 bg-accent/10 rounded-lg border border-accent/20">
              <FlaskConical size={16} className="text-accent" />
              <span className="text-text font-medium">Agente</span>
              <span className="text-xs text-text-secondary">Gera configs</span>
            </div>

            <ArrowRight size={18} className="text-text-secondary shrink-0" />

            <div className="flex flex-col items-center gap-1 px-4 py-3 bg-warning/10 rounded-lg border border-warning/20">
              <Cpu size={16} className="text-warning" />
              <span className="text-text font-medium">Treino</span>
              <span className="text-xs text-text-secondary">MPS / GPU</span>
            </div>

            <ArrowRight size={18} className="text-text-secondary shrink-0" />

            <div className="flex flex-col items-center gap-1 px-4 py-3 bg-success/10 rounded-lg border border-success/20">
              <Brain size={16} className="text-success" />
              <span className="text-text font-medium">Avaliação</span>
              <span className="text-xs text-text-secondary">Métricas + logs</span>
            </div>

            <ArrowRight size={18} className="text-text-secondary shrink-0" />

            <div className="flex flex-col items-center gap-1 px-4 py-3 bg-primary/10 rounded-lg border border-primary/20">
              <Moon size={16} className="text-primary" />
              <span className="text-text font-medium">Iteração</span>
              <span className="text-xs text-text-secondary">Loop autônomo</span>
            </div>
          </div>
        </div>
      </section>

      {/* Experiments Table */}
      <section>
        <h2 className="text-xl font-bold text-text mb-4">Experimentos</h2>
        <div className="bg-surface rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  {tableHeaders.map((header) => (
                    <th
                      key={header}
                      className="px-6 py-4 text-left font-semibold text-text-secondary"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {emptyExperiments.length === 0 ? (
                  <tr>
                    <td
                      colSpan={tableHeaders.length}
                      className="px-6 py-12 text-center text-text-secondary"
                    >
                      Nenhum experimento registrado. O treinamento será
                      iniciado na Etapa 3.
                    </td>
                  </tr>
                ) : (
                  emptyExperiments.map((exp) => (
                    <tr
                      key={exp.id}
                      className="border-b border-border last:border-0"
                    >
                      <td className="px-6 py-4 text-text font-medium">
                        {exp.modelName}
                      </td>
                      <td className="px-6 py-4 text-text tabular-nums">
                        {exp.valAccuracy != null
                          ? `${(exp.valAccuracy * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="px-6 py-4 text-text-secondary">
                        {JSON.stringify(exp.config).slice(0, 40)}...
                      </td>
                      <td className="px-6 py-4">
                        <span
                          className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            exp.status === 'completed'
                              ? 'bg-success/15 text-success'
                              : exp.status === 'running'
                                ? 'bg-primary/15 text-primary'
                                : exp.status === 'failed'
                                  ? 'bg-error/15 text-error'
                                  : 'bg-surface-light text-text-secondary'
                          }`}
                        >
                          {exp.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-text-secondary">
                        {exp.createdAt
                          ? new Date(exp.createdAt).toLocaleDateString('pt-BR')
                          : '—'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
