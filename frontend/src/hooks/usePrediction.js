import { useState, useCallback } from "react";
import { predictDisease } from "../utils/api";

export function usePrediction() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = useCallback(async (imageFile, patientInfo) => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictDisease(imageFile, patientInfo);
      setData(result);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, loading, error, predict, reset };
}
