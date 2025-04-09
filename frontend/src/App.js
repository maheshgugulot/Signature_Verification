import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import Upload from "./components/Upload";
import Login from "./components/Login";
import Register from "./components/Register";
import ProtectedRoute from "./components/ProtectedRoute";

function LogoutButton({ onLogout }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("token"); 
    onLogout(); 
    navigate("/"); 
  };

  return <button onClick={handleLogout} style={{ marginLeft: "1090px", marginTop: "10px"}}>Logout</button>;
}

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem("token"));

  useEffect(() => {
    const handleStorageChange = () => {
      setIsLoggedIn(!!localStorage.getItem("token"));
    };

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  return (
    <Router>
      <div className="app-container">
        <div className="header">
        <h1 style={{ fontSize: "2rem", marginBottom: "2rem", textAlign: "center", marginLeft: "90px"}}>Signature Verification</h1>
          {isLoggedIn && <LogoutButton onLogout={() => setIsLoggedIn(false)} />}
        </div>
        <div className="content">
          <Routes>
            <Route path="/" element={<Login setIsLoggedIn={setIsLoggedIn} />} />
            <Route path="/register" element={<Register />} />
            <Route
              path="/test"
              element={
                <ProtectedRoute>
                  <Upload />
                </ProtectedRoute>
              }
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
