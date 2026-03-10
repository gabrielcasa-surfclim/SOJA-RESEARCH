import { useState } from 'react';
import { ImageIcon, CheckCircle, Filter } from 'lucide-react';
import type { Disease, DiseaseImage } from '../types';

interface ImageGalleryProps {
  images: DiseaseImage[];
  diseases: Disease[];
}

const sourceLabels: Record<DiseaseImage['source'], string> = {
  plantvillage: 'PlantVillage',
  kaggle: 'Kaggle',
  field: 'Campo',
  upload: 'Upload',
};

const sourceColors: Record<DiseaseImage['source'], string> = {
  plantvillage: 'bg-green-500/15 text-green-400',
  kaggle: 'bg-blue-500/15 text-blue-400',
  field: 'bg-amber-500/15 text-amber-400',
  upload: 'bg-purple-500/15 text-purple-400',
};

export default function ImageGallery({ images, diseases }: ImageGalleryProps) {
  const [sourceFilter, setSourceFilter] = useState<DiseaseImage['source'] | 'all'>('all');
  const [validatedFilter, setValidatedFilter] = useState<'all' | 'validated' | 'pending'>('all');

  const filtered = images.filter((img) => {
    if (sourceFilter !== 'all' && img.source !== sourceFilter) return false;
    if (validatedFilter === 'validated' && !img.validated) return false;
    if (validatedFilter === 'pending' && img.validated) return false;
    return true;
  });

  const getDiseaseName = (diseaseId: string) =>
    diseases.find((d) => d.id === diseaseId)?.name ?? diseaseId;

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3">
        <Filter size={16} className="text-text-secondary" />

        <select
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value as DiseaseImage['source'] | 'all')}
          className="px-3 py-2 bg-surface-light border border-border rounded-lg text-sm text-text focus:outline-none focus:border-primary"
        >
          <option value="all">Todas as fontes</option>
          <option value="plantvillage">PlantVillage</option>
          <option value="kaggle">Kaggle</option>
          <option value="field">Campo</option>
          <option value="upload">Upload</option>
        </select>

        <select
          value={validatedFilter}
          onChange={(e) => setValidatedFilter(e.target.value as 'all' | 'validated' | 'pending')}
          className="px-3 py-2 bg-surface-light border border-border rounded-lg text-sm text-text focus:outline-none focus:border-primary"
        >
          <option value="all">Todas</option>
          <option value="validated">Validadas</option>
          <option value="pending">Pendentes</option>
        </select>

        <span className="ml-auto text-sm text-text-secondary">
          {filtered.length} imagens
        </span>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {filtered.map((image) => (
          <div
            key={image.id}
            className="bg-surface rounded-xl overflow-hidden group"
          >
            {/* Thumbnail placeholder */}
            <div className="relative aspect-square bg-surface-light flex items-center justify-center">
              <ImageIcon size={32} className="text-text-secondary/30" />

              {/* Source badge */}
              <span
                className={`absolute top-2 left-2 px-2 py-0.5 rounded-full text-xs font-medium ${sourceColors[image.source]}`}
              >
                {sourceLabels[image.source]}
              </span>

              {/* Validated checkmark */}
              {image.validated && (
                <CheckCircle
                  size={20}
                  className="absolute top-2 right-2 text-success"
                />
              )}
            </div>

            {/* Info */}
            <div className="p-3">
              <p className="text-sm font-medium text-text truncate">
                {getDiseaseName(image.diseaseId)}
              </p>
              <p className="text-xs text-text-secondary mt-0.5">
                {new Date(image.createdAt).toLocaleDateString('pt-BR')}
              </p>
            </div>
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-text-secondary">
          Nenhuma imagem encontrada com os filtros selecionados.
        </div>
      )}
    </div>
  );
}
