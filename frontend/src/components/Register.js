import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

function Register() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      setMessage("Passwords do not match");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:5000/register", {
        username,
        password,
      });

      setMessage(response.data.msg);
      setTimeout(() => navigate("/"), 2000); 
    } catch (error) {
      setMessage(error.response?.data?.msg || "Registration failed");
    }
  };

  return (
    <div className="main-content" style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "100vh" }}>
      <div className="card" style={{ width: "100%", maxWidth: "400px", padding: "2rem", boxShadow: "0 4px 8px rgba(0,0,0,0.1)", borderRadius: "8px" }}>
        <h2 style={{ textAlign: "center" }}>Register</h2>
        <form onSubmit={handleRegister}>
          <div className="form-group" style={{ marginBottom: "1rem" }}>
            <label>Username:</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              style={{ width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
            />
          </div>
          <div className="form-group" style={{ marginBottom: "1rem" }}>
            <label>Password:</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{ width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
            />
          </div>
          <div className="form-group" style={{ marginBottom: "1rem" }}>
            <label>Confirm Password:</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              style={{ width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
            />
          </div>
          {message && <p style={{ color: message.includes("success") ? "green" : "red", marginBottom: "1rem", textAlign: "center" }}>{message}</p>}
          <button type="submit" style={{ width: "100%", padding: "0.75rem", backgroundColor: "#4CAF50", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}>Register</button>
        </form>
        <p style={{ marginTop: "1rem", textAlign: "center" }}>
          Already have an account? <a href="/" className="App-link">Login</a>
        </p>
      </div>
    </div>
  );
}

export default Register;
