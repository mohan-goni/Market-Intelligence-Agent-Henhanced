import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onFilesUploaded: (fileIds: string[]) => void;
  maxFileSize?: number; // in bytes
  maxFiles?: number;
  acceptedFileTypes?: Record<string, string[]>;
}

interface UploadedFile {
  id: string;
  filename: string;
  content_type: string;
  size: number;
  upload_time: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFilesUploaded,
  maxFileSize = 50 * 1024 * 1024, // 50MB default
  maxFiles = 10,
  acceptedFileTypes = {
    'text/csv': ['.csv'],
    'application/pdf': ['.pdf'],
    'application/json': ['.json'],
    'text/plain': ['.txt'],
    'application/zip': ['.zip'],
    'audio/mpeg': ['.mp3'],
    'audio/wav': ['.wav'],
    'video/mp4': ['.mp4'],
    'video/webm': ['.webm']
  }
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Check if adding these files would exceed the max files limit
    if (files.length + acceptedFiles.length > maxFiles) {
      setError(`You can only upload a maximum of ${maxFiles} files.`);
      return;
    }

    // Filter out files that exceed the size limit
    const validFiles = acceptedFiles.filter(file => {
      if (file.size > maxFileSize) {
        setError(`File "${file.name}" exceeds the maximum file size of ${maxFileSize / (1024 * 1024)}MB.`);
        return false;
      }
      return true;
    });

    setFiles(prevFiles => [...prevFiles, ...validFiles]);
    setError(null);
  }, [files, maxFiles, maxFileSize]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    maxSize: maxFileSize,
    maxFiles: maxFiles
  });

  const removeFile = (index: number) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const uploadFiles = async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    setError(null);
    
    const uploadedFileIds: string[] = [];
    
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('file', file);
        
        // Initialize progress for this file
        setUploadProgress(prev => ({ ...prev, [file.name]: 0 }));
        
        const xhr = new XMLHttpRequest();
        
        // Set up progress tracking
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const percentComplete = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(prev => ({ ...prev, [file.name]: percentComplete }));
          }
        });
        
        // Create a promise to handle the XHR request
        const uploadPromise = new Promise<UploadedFile>((resolve, reject) => {
          xhr.open('POST', '/upload-file');
          xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('token')}`);
          
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              const response = JSON.parse(xhr.responseText);
              resolve({
                id: response.file_id,
                filename: response.filename,
                content_type: response.content_type,
                size: response.size,
                upload_time: response.upload_time
              });
            } else {
              reject(new Error(`Upload failed with status ${xhr.status}`));
            }
          };
          
          xhr.onerror = () => {
            reject(new Error('Network error occurred during upload'));
          };
          
          xhr.send(formData);
        });
        
        // Wait for the upload to complete
        const uploadedFile = await uploadPromise;
        uploadedFileIds.push(uploadedFile.id);
        setUploadedFiles(prev => [...prev, uploadedFile]);
      }
      
      // Clear the files list after successful upload
      setFiles([]);
      
      // Notify parent component of uploaded file IDs
      onFilesUploaded(uploadedFileIds);
      
    } catch (err) {
      console.error('Error uploading files:', err);
      setError('Failed to upload one or more files. Please try again.');
    } finally {
      setIsUploading(false);
      setUploadProgress({});
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-md p-6 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20'
            : 'border-gray-300 dark:border-gray-600 hover:border-indigo-400 dark:hover:border-indigo-500'
        }`}
      >
        <input {...getInputProps()} />
        <div className="space-y-2">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            stroke="currentColor"
            fill="none"
            viewBox="0 0 48 48"
            aria-hidden="true"
          >
            <path
              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <div className="flex text-sm text-gray-600 dark:text-gray-400 justify-center">
            <span className="relative font-medium text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
              Upload files
            </span>
            <p className="pl-1">or drag and drop</p>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            CSV, PDF, JSON, TXT, ZIP, MP3, MP4 up to {maxFileSize / (1024 * 1024)}MB each
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">{error}</span>
        </div>
      )}

      {files.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Selected Files:</h4>
          <ul className="space-y-2 max-h-60 overflow-y-auto">
            {files.map((file, index) => (
              <li key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded-md">
                <div className="flex items-center space-x-2 truncate max-w-xs">
                  <span className="text-sm text-gray-600 dark:text-gray-300 truncate">{file.name}</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">({formatFileSize(file.size)})</span>
                </div>
                {isUploading && uploadProgress[file.name] !== undefined ? (
                  <div className="w-20">
                    <div className="bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                      <div
                        className="bg-indigo-600 h-2 rounded-full"
                        style={{ width: `${uploadProgress[file.name]}%` }}
                      ></div>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => removeFile(index)}
                    className="text-red-500 hover:text-red-700"
                    disabled={isUploading}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                )}
              </li>
            ))}
          </ul>

          <button
            type="button"
            onClick={uploadFiles}
            disabled={isUploading || files.length === 0}
            className={`mt-4 w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              isUploading || files.length === 0
                ? 'bg-indigo-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
            }`}
          >
            {isUploading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Uploading...
              </>
            ) : (
              'Upload Files'
            )}
          </button>
        </div>
      )}

      {uploadedFiles.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Uploaded Files:</h4>
          <ul className="space-y-2 max-h-60 overflow-y-auto">
            {uploadedFiles.map((file, index) => (
              <li key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded-md">
                <div className="flex items-center space-x-2 truncate max-w-xs">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-500" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span className="text-sm text-gray-600 dark:text-gray-300 truncate">{file.filename}</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">({formatFileSize(file.size)})</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
