import { describe, expect, it } from 'vitest';
import { byTask } from './api';

describe('task order', () => {
  it('puts 2 before 10, and named task ids in alphabetical order', () => {
    expect(['10', '2', '0', '1', '19'].sort(byTask)).toEqual(['0', '1', '2', '10', '19']);
    expect(['[mobile_data]b', '[mobile_data]a'].sort(byTask)).toEqual(['[mobile_data]a', '[mobile_data]b']);
  });
});
