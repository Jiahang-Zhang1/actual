import React from 'react';

export function AISuggestionBadge({
  confidence,
}: {
  confidence: number;
}) {
  return (
    <span
      style={{
        marginLeft: 6,
        padding: '2px 6px',
        borderRadius: 10,
        fontSize: 11,
        background: '#2b5cff22',
      }}
      title={`AI suggestion (${Math.round(confidence * 100)}%)`}
    >
      AI {Math.round(confidence * 100)}%
    </span>
  );
}
