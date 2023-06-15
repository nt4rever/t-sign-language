import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import React, { useRef, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { IMAGE_CHANGE, REPORT_CHANGE, RES_CHANGE } from "../../store/actions";
import API from "../../utils/API.js";
import "./main.css";

function Main() {
  const fileRef = useRef(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [testImage, setTestImage] = useState(null);
  const { image } = useSelector((state) => state.image);
  const { res } = useSelector((state) => state.res);
  const dispatch = useDispatch();

  const handleUploadClick = () => {
    fileRef.current.click();
  };

  const handlePredictClick = async () => {
    await API.post("/predict", { uri: image }).then((response) => {
      if (response.status === 200) {
        dispatch({
          type: RES_CHANGE,
          res: response.data.result,
        });
      } else return;
    });
  };

  const handleChange = async (e) => {
    setSelectedImage(e.target.files[0]);
    const base64 = await convertBase64(e.target.files[0]);
    dispatch({
      type: IMAGE_CHANGE,
      image: base64,
    });
  };

  const convertBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(file);
      fileReader.onload = () => {
        resolve(fileReader.result);
      };
      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  };

  const handleTestClick = async () => {
    await API.post("/test", { uri: image }).then((response) => {
      if (response.status === 200) {
        setTestImage(response.data.image);
      } else return;
    });
  };

  const handleReport = async () => {
    dispatch({
      type: REPORT_CHANGE,
      open: true,
    });
    await API.get("/report").then((response) => {
      if (response.status === 200) {
        dispatch({
          type: REPORT_CHANGE,
          data: response.data.data,
          cm: response.data.cm,
        });
      } else return;
    });
  };

  const handleFull = () => {
    window.open("http://localhost:8000/gestures");
  };

  return (
    <div className="container">
      <div className="main">
        <div className="input-img">
          {selectedImage && (
            <img
              src={URL.createObjectURL(selectedImage)}
              width="200px"
              alt="test"
            />
          )}
          {testImage && (
            <img
              src={"data:image/png;charset=utf-8;base64," + testImage}
              width="200px"
              alt="test"
            />
          )}
        </div>
        <Box
          component="span"
          sx={{
            p: 1,
            border: "1px dashed grey",
            backgroundColor: "#fff",
            borderRadius: "10px",
          }}
        >
          <Stack direction="row" justifyContent="center" spacing={2}>
            <Button variant="outlined" onClick={handleUploadClick}>
              Upload
              <input
                ref={fileRef}
                className="imageUpload"
                type="file"
                accept="image/*"
                onChange={handleChange}
              />
            </Button>
            {selectedImage && (
              <Button
                variant="contained"
                color="secondary"
                onClick={handleTestClick}
              >
                Test
              </Button>
            )}

            <Button
              variant="contained"
              color="success"
              onClick={handlePredictClick}
            >
              Predict
            </Button>
            {res && <Chip label={"Result: " + res} color="primary" />}
          </Stack>
          <Stack
            direction="row"
            spacing={2}
            sx={{ mt: 2, justifyContent: "center" }}
          >
            <Button variant="outlined" color="error" onClick={handleReport}>
              Report
            </Button>
            <Button variant="outlined" color="primary" onClick={handleFull}>
              Full Gesture
            </Button>
          </Stack>
        </Box>
      </div>
    </div>
  );
}

export default Main;
