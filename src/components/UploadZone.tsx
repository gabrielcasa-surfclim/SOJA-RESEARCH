import { useState, useRef, useCallback } from 'react';
import { Upload, ImagePlus } from 'lucide-react';

interface UploadZoneProps {
  onUpload: (files: File[]) => void;
  diseaseId?: string;
}

export default function UploadZone({ onUpload, diseaseId: _diseaseId }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedCount, setSelectedCount] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const imageFiles = Array.from(files).filter((f) =>
        f.type.startsWith('image/')
      );
      if (imageFiles.length > 0) {
        setSelectedCount(imageFiles.length);
        onUpload(imageFiles);
      }
    },
    [onUpload]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') handleClick();
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`flex flex-col items-center justify-center gap-4 p-12 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
        isDragging
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/50 hover:bg-surface-light'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={(e) => handleFiles(e.target.files)}
        className="hidden"
      />

      <div className="p-4 rounded-full bg-primary/10">
        {isDragging ? (
          <ImagePlus size={32} className="text-primary" />
        ) : (
          <Upload size={32} className="text-primary" />
        )}
      </div>

      <div className="text-center">
        <p className="text-text font-medium">
          Arraste imagens ou clique para selecionar
        </p>
        <p className="text-sm text-text-secondary mt-1">
          Formatos aceitos: PNG, JPG, WEBP
        </p>
      </div>

      {selectedCount > 0 && (
        <p className="text-sm text-primary font-medium">
          {selectedCount} {selectedCount === 1 ? 'arquivo selecionado' : 'arquivos selecionados'}
        </p>
      )}
    </div>
  );
}
