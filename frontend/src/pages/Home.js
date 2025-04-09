import { useState } from "react";
import Upload from "../components/Upload";
import Result from "../components/Result";

const Home = () => {
  const [verificationResult, setVerificationResult] = useState(null);

  return (
    <div className="main-content">
      <div className="card">
        <h1 style={{ fontSize: "2rem", marginBottom: "2rem", textAlign: "center"}}>Signature Verification</h1>
        <Upload onUpload={setVerificationResult} />
        {verificationResult && <Result result={verificationResult} />}
      </div>
    </div>
  );
};

export default Home;
