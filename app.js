// apt-get update && apt-get install -y curl
// url -sL https://deb.nodesource.com/setup_16.x | bash -
// apt-get install -y nodejs
// npx create-react-app stl-viewer
// cd stl-viewer
// curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.4/install.sh | bash
// source ~/.bashrc
// source ~/.profile
// source ~/.nvmrc
// nvm --version

import React, { Suspense, useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";

function Model({ url }) {
  const [model, setModel] = useState(null);

  useEffect(() => {
    const loader = new STLLoader();
    loader.load(url, (geometry) => {
      console.log("STL загружен:", geometry);
      geometry.computeVertexNormals();
      setModel(geometry);
    });
  }, [url]);

  return model ? (
    <mesh geometry={model} scale={0.01} position={[0, 0, 0]}>
      <meshStandardMaterial color="yellow" />
    </mesh>
  ) : null;
}

function App() {
  return (
    <div style={{ height: "100vh", margin: 0 }}>
      <Canvas>
        <Suspense fallback={null}>
          <ambientLight intensity={0.5} />
          <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
          <Model url="/models/SquareTrussStraightSegment_21_Stl.STL" />
        </Suspense>
        <OrbitControls />
      </Canvas>
    </div>
  );
}

export default App;
