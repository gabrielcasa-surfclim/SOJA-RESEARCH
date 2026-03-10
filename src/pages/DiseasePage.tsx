import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Eye,
  Images,
  FileText,
  AlertTriangle,
  Leaf,
  Droplets,
  Shield,
  FlaskConical,
} from 'lucide-react';
import { diseases } from '../data/diseases';
import type { DiseaseImage } from '../types';
import ImageGallery from '../components/ImageGallery';

const severityConfig = {
  alta: { label: 'Alta', className: 'bg-error/15 text-error' },
  média: { label: 'Média', className: 'bg-accent/15 text-accent' },
  baixa: { label: 'Baixa', className: 'bg-success/15 text-success' },
};

type Tab = 'overview' | 'gallery' | 'articles';

const tabs: { id: Tab; label: string; icon: typeof Eye }[] = [
  { id: 'overview', label: 'Visão Geral', icon: Eye },
  { id: 'gallery', label: 'Galeria', icon: Images },
  { id: 'articles', label: 'Artigos', icon: FileText },
];

export default function DiseasePage() {
  const { diseaseId } = useParams<{ diseaseId: string }>();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<Tab>('overview');

  const disease = diseases.find((d) => d.id === diseaseId);

  if (!disease) {
    return (
      <div className="text-center py-20">
        <p className="text-text-secondary text-lg">Doença não encontrada.</p>
        <button
          onClick={() => navigate('/')}
          className="mt-4 px-4 py-2 bg-primary text-bg rounded-lg hover:bg-primary-light transition-colors"
        >
          Voltar ao Dashboard
        </button>
      </div>
    );
  }

  const severity = severityConfig[disease.severity];
  const mockImages: DiseaseImage[] = [];

  return (
    <div className="space-y-6">
      {/* Back button */}
      <button
        onClick={() => navigate('/')}
        className="flex items-center gap-2 text-text-secondary hover:text-text transition-colors"
      >
        <ArrowLeft size={18} />
        <span className="text-sm">Voltar ao Dashboard</span>
      </button>

      {/* Hero Section */}
      <div className="bg-surface rounded-xl p-8">
        <div className="flex items-start gap-6">
          <span className="text-6xl">{disease.icon}</span>
          <div className="flex-1">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-3xl font-bold text-text">{disease.name}</h1>
              <span
                className={`px-3 py-1 rounded-full text-sm font-medium ${severity.className}`}
              >
                Severidade: {severity.label}
              </span>
            </div>
            <p className="mt-2 text-lg italic text-text-secondary">
              {disease.scientificName}
            </p>
            <p className="mt-3 text-sm text-text-secondary">
              {disease.imageCount} imagens &middot; {disease.articleCount}{' '}
              artigos
            </p>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 bg-surface rounded-xl p-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-primary text-bg'
                  : 'text-text-secondary hover:text-text hover:bg-surface-light'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Symptoms */}
          <div className="bg-surface rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle size={18} className="text-warning" />
              <h3 className="text-lg font-semibold text-text">Sintomas</h3>
            </div>
            <ul className="space-y-2">
              {disease.symptoms.map((symptom) => (
                <li
                  key={symptom}
                  className="flex items-start gap-2 text-sm text-text-secondary"
                >
                  <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-warning shrink-0" />
                  {symptom}
                </li>
              ))}
            </ul>
          </div>

          {/* Conditions */}
          <div className="bg-surface rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Droplets size={18} className="text-primary" />
              <h3 className="text-lg font-semibold text-text">
                Condições Favoráveis
              </h3>
            </div>
            <p className="text-sm text-text-secondary leading-relaxed">
              {disease.conditions}
            </p>
          </div>

          {/* Management */}
          <div className="bg-surface rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Shield size={18} className="text-success" />
              <h3 className="text-lg font-semibold text-text">Manejo</h3>
            </div>
            <ul className="space-y-2">
              {disease.management.map((item) => (
                <li
                  key={item}
                  className="flex items-start gap-2 text-sm text-text-secondary"
                >
                  <Leaf size={14} className="mt-0.5 text-success shrink-0" />
                  {item}
                </li>
              ))}
            </ul>
          </div>

          {/* Fungicides */}
          <div className="bg-surface rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <FlaskConical size={18} className="text-accent" />
              <h3 className="text-lg font-semibold text-text">Fungicidas</h3>
            </div>
            <ul className="space-y-2">
              {disease.fungicides.map((fungicide) => (
                <li
                  key={fungicide}
                  className="flex items-start gap-2 text-sm text-text-secondary"
                >
                  <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
                  {fungicide}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {activeTab === 'gallery' && (
        <ImageGallery images={mockImages} diseases={diseases} />
      )}

      {activeTab === 'articles' && (
        <div className="bg-surface rounded-xl p-12 text-center">
          <FileText size={48} className="mx-auto text-text-secondary/30" />
          <p className="mt-4 text-text-secondary">
            Busque artigos na página de{' '}
            <button
              onClick={() => navigate('/pesquisa')}
              className="text-primary hover:text-primary-light underline transition-colors"
            >
              Pesquisa
            </button>
          </p>
        </div>
      )}
    </div>
  );
}
