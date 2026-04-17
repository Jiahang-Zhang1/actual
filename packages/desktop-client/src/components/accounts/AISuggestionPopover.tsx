import React from 'react';

type Suggestion = {
  category_id: string;
  score: number;
};

export function AISuggestionPopover({
  topCategories,
  onAccept,
}: {
  topCategories: Suggestion[];
  onAccept: (categoryId: string) => void;
}) {
  return (
    <div style={{ padding: 8, minWidth: 220 }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>AI suggestions</div>
      {topCategories.map(item => (
        <button
          key={item.category_id}
          onClick={() => onAccept(item.category_id)}
          style={{
            display: 'block',
            width: '100%',
            textAlign: 'left',
            marginBottom: 6,
          }}
        >
          {item.category_id} ({Math.round(item.score * 100)}%)
        </button>
      ))}
    </div>
  );
}
