// State
let cvReady = false;
let currentImageData = null; // original HTMLImageElement

// DOM elements
const cameraBtn = document.getElementById('cameraBtn');
const galleryBtn = document.getElementById('galleryBtn');
const cameraInput = document.getElementById('cameraInput');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const seedCountEl = document.getElementById('seedCount');
const originalCanvas = document.getElementById('originalCanvas');
const resultCanvas = document.getElementById('resultCanvas');
const threshCanvas = document.getElementById('threshCanvas');

// Parameter elements
const blurSize = document.getElementById('blurSize');
const minArea = document.getElementById('minArea');
const maxArea = document.getElementById('maxArea');
const kernelSize = document.getElementById('kernelSize');
const invertToggle = document.getElementById('invertToggle');
const separationLevel = document.getElementById('separationLevel');
const minCircularity = document.getElementById('minCircularity');

const blurVal = document.getElementById('blurVal');
const minAreaVal = document.getElementById('minAreaVal');
const maxAreaVal = document.getElementById('maxAreaVal');
const kernelVal = document.getElementById('kernelVal');
const sepVal = document.getElementById('sepVal');
const circVal = document.getElementById('circVal');

// OpenCV ready callback
function onOpenCvReady() {
  cvReady = true;
  loading.classList.add('hidden');
  console.log('OpenCV.js is ready');
  if (currentImageData) {
    processImage();
  }
}

// Make it global for the script onload
window.onOpenCvReady = onOpenCvReady;

// --- File Upload ---
cameraBtn.addEventListener('click', () => cameraInput.click());
galleryBtn.addEventListener('click', () => fileInput.click());

cameraInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      currentImageData = img;
      // Show the result section
      resultSection.classList.remove('hidden');
      // Draw original
      drawOriginal(img);
      if (cvReady) {
        processImage();
      } else {
        loading.classList.remove('hidden');
      }
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function drawOriginal(img) {
  originalCanvas.width = img.width;
  originalCanvas.height = img.height;
  const ctx = originalCanvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
}

// --- Parameter change handlers ---
function updateParamDisplay() {
  blurVal.textContent = blurSize.value;
  minAreaVal.textContent = minArea.value;
  maxAreaVal.textContent = maxArea.value;
  kernelVal.textContent = kernelSize.value;
  sepVal.textContent = separationLevel.value;
  circVal.textContent = minCircularity.value;
}

[blurSize, minArea, maxArea, kernelSize, invertToggle, separationLevel, minCircularity].forEach(el => {
  el.addEventListener('input', () => {
    updateParamDisplay();
    if (currentImageData && cvReady) {
      processImage();
    }
  });
});

// --- Core: Seed Counting with OpenCV.js ---
function processImage() {
  if (!currentImageData || !cvReady) return;

  const img = currentImageData;

  // Read parameters
  const blur = parseInt(blurSize.value);
  const minA = parseInt(minArea.value);
  const maxA = parseInt(maxArea.value);
  const kSize = parseInt(kernelSize.value);
  const invert = invertToggle.checked;
  const sepLevel = parseInt(separationLevel.value) / 100; // 0 to 0.8
  const minCirc = parseInt(minCircularity.value) / 100;   // 0 to 1

  // Load image into OpenCV Mat from the original canvas
  drawOriginal(img);
  const src = cv.imread(originalCanvas);
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const thresh = new cv.Mat();
  const morphed = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  // Track extra mats for cleanup
  const extraMats = [];

  try {
    // 1. Convert to grayscale
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // 2. Gaussian blur
    const ksize = new cv.Size(blur, blur);
    cv.GaussianBlur(gray, blurred, ksize, 0);

    // 3. Otsu thresholding
    if (invert) {
      cv.threshold(blurred, thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    } else {
      cv.threshold(blurred, thresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    }

    // 4. Morphological opening to remove noise, then closing to fill gaps
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kSize, kSize));
    cv.morphologyEx(thresh, morphed, cv.MORPH_OPEN, kernel);
    cv.morphologyEx(morphed, morphed, cv.MORPH_CLOSE, kernel);
    kernel.delete();

    // 5. Distance transform separation for touching seeds
    if (sepLevel > 0) {
      const dist = new cv.Mat();
      extraMats.push(dist);
      cv.distanceTransform(morphed, dist, cv.DIST_L2, 5);
      cv.normalize(dist, dist, 0, 1, cv.NORM_MINMAX);

      // Threshold to get seed centers
      const separated = new cv.Mat();
      extraMats.push(separated);
      cv.threshold(dist, separated, sepLevel, 1, cv.THRESH_BINARY);
      separated.convertTo(separated, cv.CV_8U, 255);

      // Dilate the separated centers back, constrained to the original morphed mask,
      // so area values stay meaningful for filtering
      const dilKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
      const dilated = new cv.Mat();
      extraMats.push(dilated);
      cv.dilate(separated, dilated, dilKernel, new cv.Point(-1, -1), 3);
      cv.bitwise_and(dilated, morphed, morphed);
      dilKernel.delete();
    }

    // Show threshold preview
    cv.imshow(threshCanvas, morphed);

    // 6. Find contours
    cv.findContours(morphed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // 7. Filter contours by area and circularity, then draw
    const result = src.clone();
    let count = 0;

    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);

      if (area < minA || area > maxA) continue;

      // Circularity filter: circularity = 4pi * area / perimeter^2
      // Perfect circle = 1.0, square ~ 0.785, very elongated < 0.3
      if (minCirc > 0) {
        const perimeter = cv.arcLength(cnt, true);
        if (perimeter > 0) {
          const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
          if (circularity < minCirc) continue;
        }
      }

      count++;

      // Draw contour
      cv.drawContours(result, contours, i, new cv.Scalar(0, 255, 0, 255), 2);

      // Draw number label
      const moments = cv.moments(cnt);
      if (moments.m00 !== 0) {
        const cx = Math.round(moments.m10 / moments.m00);
        const cy = Math.round(moments.m01 / moments.m00);
        const fontSize = Math.max(0.4, Math.min(1.2, Math.sqrt(area) / 40));
        // Background for text readability
        cv.putText(result, String(count), new cv.Point(cx - 5, cy + 5),
          cv.FONT_HERSHEY_SIMPLEX, fontSize, new cv.Scalar(0, 0, 0, 255), 3);
        cv.putText(result, String(count), new cv.Point(cx - 5, cy + 5),
          cv.FONT_HERSHEY_SIMPLEX, fontSize, new cv.Scalar(255, 255, 0, 255), 1);
      }
    }

    // Display result
    cv.imshow(resultCanvas, result);
    seedCountEl.textContent = count;

    result.delete();
  } catch (err) {
    console.error('Processing error:', err);
    seedCountEl.textContent = 'Error';
  } finally {
    // Clean up
    src.delete();
    gray.delete();
    blurred.delete();
    thresh.delete();
    morphed.delete();
    contours.delete();
    hierarchy.delete();
    extraMats.forEach(m => m.delete());
  }
}
