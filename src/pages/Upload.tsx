import { useState } from 'react';
import { Upload as UploadIcon, Info, Send } from 'lucide-react';
import { diseases } from '../data/diseases';
import UploadZone from '../components/UploadZone';

type Source = 'plantvillage' | 'kaggle' | 'field' | 'upload';
type Stage = 'inicial' | 'avançado';

export default function Upload() {
  const [selectedDisease, setSelectedDisease] = useState('');
  const [source, setSource] = useState<Source>('upload');
  const [stage, setStage] = useState<Stage>('inicial');
  const [notes, setNotes] = useState('');
  const [files, setFiles] = useState<File[]>([]);

  const handleUpload = (uploadedFiles: File[]) => {
    setFiles(uploadedFiles);
  };

  const handleSubmit = () => {
    if (!selectedDisease) {
      alert('Selecione uma doença antes de enviar.');
      return;
    }
    if (files.length === 0) {
      alert('Selecione pelo menos uma imagem.');
      return;
    }
    alert(
      `Upload simulado!\n\nDoença: ${selectedDisease}\nFonte: ${source}\nEstágio: ${stage}\nImagens: ${files.length}\nNotas: ${notes || '(nenhuma)'}\n\nSupabase não conectado — dados não foram salvos.`
    );
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <UploadIcon size={24} className="text-primary" />
          <h1 className="text-2xl font-bold text-text">Upload de Imagens</h1>
        </div>
        <p className="mt-1 text-text-secondary">
          Contribua com imagens para expandir o dataset de treinamento.
        </p>
      </div>

      {/* Disease Selector */}
      <div className="bg-surface rounded-xl p-6 space-y-4">
        <label className="block">
          <span className="text-sm font-medium text-text">
            Doença identificada *
          </span>
          <select
            value={selectedDisease}
            onChange={(e) => setSelectedDisease(e.target.value)}
            className="mt-1.5 w-full px-4 py-3 bg-surface-light border border-border rounded-xl text-text focus:outline-none focus:border-primary transition-colors"
          >
            <option value="">Selecione uma doença...</option>
            {diseases.map((d) => (
              <option key={d.id} value={d.id}>
                {d.icon} {d.name} — {d.scientificName}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Upload Zone */}
      <UploadZone onUpload={handleUpload} diseaseId={selectedDisease} />

      {/* Metadata Form */}
      <div className="bg-surface rounded-xl p-6 space-y-5">
        <h2 className="text-lg font-semibold text-text">Metadados</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <label className="block">
            <span className="text-sm font-medium text-text">Fonte</span>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value as Source)}
              className="mt-1.5 w-full px-4 py-3 bg-surface-light border border-border rounded-xl text-text focus:outline-none focus:border-primary transition-colors"
            >
              <option value="upload">Upload manual</option>
              <option value="field">Campo (foto própria)</option>
              <option value="plantvillage">PlantVillage</option>
              <option value="kaggle">Kaggle</option>
            </select>
          </label>

          <label className="block">
            <span className="text-sm font-medium text-text">
              Estágio da doença
            </span>
            <select
              value={stage}
              onChange={(e) => setStage(e.target.value as Stage)}
              className="mt-1.5 w-full px-4 py-3 bg-surface-light border border-border rounded-xl text-text focus:outline-none focus:border-primary transition-colors"
            >
              <option value="inicial">Inicial</option>
              <option value="avançado">Avançado</option>
            </select>
          </label>
        </div>

        <label className="block">
          <span className="text-sm font-medium text-text">
            Observações (opcional)
          </span>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder="Condições de captura, localização, cultivar..."
            className="mt-1.5 w-full px-4 py-3 bg-surface-light border border-border rounded-xl text-text placeholder:text-text-secondary/50 focus:outline-none focus:border-primary transition-colors resize-none"
          />
        </label>
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        className="w-full flex items-center justify-center gap-2 px-6 py-3.5 bg-primary hover:bg-primary-light text-bg font-semibold rounded-xl transition-colors"
      >
        <Send size={18} />
        Enviar Imagens
      </button>

      {/* Instructions */}
      <div className="bg-surface rounded-xl p-6">
        <div className="flex items-center gap-2 mb-3">
          <Info size={18} className="text-primary" />
          <h3 className="text-sm font-semibold text-text">
            Requisitos de Imagem
          </h3>
        </div>
        <ul className="space-y-1.5 text-sm text-text-secondary">
          <li>
            &bull; Formatos aceitos: PNG, JPG ou WEBP
          </li>
          <li>
            &bull; Resolução mínima recomendada: 224x224 pixels
          </li>
          <li>
            &bull; Foco nítido na folha ou região afetada
          </li>
          <li>
            &bull; Evite imagens com marca d&apos;água ou texto sobreposto
          </li>
          <li>
            &bull; Uma doença por imagem para melhor classificação
          </li>
          <li>
            &bull; Imagens de campo são especialmente valiosas para diversidade do dataset
          </li>
        </ul>
      </div>
    </div>
  );
}
