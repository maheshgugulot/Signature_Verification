const Result = ({ result }) => {
    return (
      <div className="p-4 border rounded-lg shadow-md" style={{ 
        display: "flex", 
        flexDirection: "column", 
        alignItems: "center", 
        justifyContent: "center",
        maxWidth: "600px",
        margin: "0 auto",
        padding: "1.5rem",
        boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
        borderRadius: "8px"
      }}>
        <h2 className="text-xl font-bold" style={{ textAlign: "center", marginBottom: "1rem" }}>Verification Result</h2>
        <p className="text-gray-700" style={{ textAlign: "center" }}>{result}</p>
      </div>
    );
  };
  
  export default Result;
  