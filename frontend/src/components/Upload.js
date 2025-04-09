import React, { useState } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Upload = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      toast.error("Please select a file!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setPredictions(response.data);
      toast.success("Prediction Successful!");
    } catch (error) {
      toast.error("Error processing the image!");
    }
  };

  return (
    <div className="main-content" style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh", padding: "2rem" }}>
      <div className="card" style={{ width: "100%", maxWidth: "600px", padding: "2rem", boxShadow: "0 4px 8px rgba(0,0,0,0.1)", borderRadius: "8px" }}>
        <h2 style={{ textAlign: "center", marginBottom: "1.5rem" }}>Upload Signature Image</h2>
        <div className="form-group" style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: "1.5rem" }}>
          <input 
            type="file" 
            onChange={handleFileChange} 
            accept="image/*"
            style={{ marginBottom: "1rem", width: "100%" }}
          />
          <button 
            onClick={handleUpload}
            style={{ 
              padding: "0.75rem 1.5rem", 
              backgroundColor: "#4CAF50", 
              color: "white", 
              border: "none", 
              borderRadius: "4px", 
              cursor: "pointer",
              width: "100%",
              maxWidth: "200px"
            }}
          >
            Upload
          </button>
        </div>

        {predictions && (
          <div className="card" style={{ marginTop: "2rem", padding: "1.5rem", boxShadow: "0 2px 4px rgba(0,0,0,0.1)", borderRadius: "8px" }}>
            <h3 style={{ textAlign: "center", marginBottom: "1rem" }}>Predictions:</h3>
            <div className="form-group" style={{ textAlign: "center" }}>
              <p><b>SVM:</b> {predictions.SVM}</p>
              <p><b>KNN:</b> {predictions.KNN}</p>
              <p><b>LSTM:</b> {predictions.LSTM}</p>
              <p><b>Similarity Score:</b> {predictions.similarity_score}</p>
            </div>
          </div>
        )}

        {predictions?.visualization && (
          <div className="card" style={{ marginTop: "2rem", padding: "1.5rem", boxShadow: "0 2px 4px rgba(0,0,0,0.1)", borderRadius: "8px", textAlign: "center" }}>
            <h3 style={{ marginBottom: "1rem" }}>Visualized Differences:</h3>
            <img
              src={`data:image/png;base64,${predictions.visualization}`}
              alt="Differences"
              style={{ 
                maxWidth: "100%", 
                height: "auto",
                borderRadius: "4px",
                boxShadow: "0 2px 4px rgba(0,0,0,0.1)"
              }}
            />
          </div>
        )}

        <ToastContainer />
      </div>
    </div>
  );
};

export default Upload;
