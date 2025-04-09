import React from "react";
import { useNavigate } from "react-router-dom";

function Logout() {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("token"); 
    navigate("/"); 
  };

  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
      <button 
        onClick={handleLogout}
        style={{ 
          padding: "0.75rem 1.5rem", 
          backgroundColor: "#f44336", 
          color: "white", 
          border: "none", 
          borderRadius: "4px", 
          cursor: "pointer"
        }}
      >
        Logout
      </button>
    </div>
  );
}

export default Logout;
