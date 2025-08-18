import React from 'react';
import { CAT_COLORS } from '@/lib/constants';

export default function ColorChip({ category }){
  return <span className={`px-2 py-0.5 rounded-md text-[11px] ${CAT_COLORS[category]?.chip || CAT_COLORS.Meta.chip}`}>{category}</span>;
}
