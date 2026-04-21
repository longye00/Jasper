// app.js

let cvReady = false; // 标记 OpenCV 是否就绪
let srcImage = null; // 缓存原始图片矩阵

// 1. 立即初始化 UI 事件（即使 OpenCV 还没加载完，按钮也要能点）
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  // 兼容性检查：如果 opencv.js 已经提前加载好了
  if (typeof cv !== 'undefined' && cv.Mat) {
    onOpenCvReady();
  }
});

// 2. OpenCV 加载后的回调
function onOpenCvReady() {
  console.log('OpenCV.js 已就绪');
  cvReady = true;
  const loadingEl = document.getElementById('loading');
  if (loadingEl) loadingEl.classList.add('hidden');
  
  // 如果用户在加载完成前已经选好了图片，立即触发一次处理
  if (document.getElementById('originalCanvas').width > 0) {
    processImage();
  }
}

function setupEventListeners() {
  const cameraInput = document.getElementById('cameraInput');
  const fileInput = document.getElementById('fileInput');
  const cameraBtn = document.getElementById('cameraBtn');
  const galleryBtn = document.getElementById('galleryBtn');

  // 绑定点击事件：立即弹出系统相机/相册
  if (cameraBtn) cameraBtn.addEventListener('click', () => cameraInput.click());
  if (galleryBtn) galleryBtn.addEventListener('click', () => fileInput.click());

  cameraInput.addEventListener('change', handleImageUpload);
  fileInput.addEventListener('change', handleImageUpload);

  // 参数滑块实时联动
  const params = ['blurSize', 'minArea', 'maxArea', 'kernelSize', 'separationLevel', 'minCircularity'];
  params.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', (e) => {
        const valId = id.replace('Size', 'Val').replace('Level', 'Val').replace('minCircularity', 'circVal');
        const valEl = document.getElementById(valId);
        if (valEl) valEl.innerText = e.target.value;
        if (cvReady) processImage();
      });
    }
  });

  const invertToggle = document.getElementById('invertToggle');
  if (invertToggle) invertToggle.addEventListener('change', () => {
    if (cvReady) processImage();
  });
}

function handleImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.getElementById('originalCanvas');
      // 限制最大处理宽度为 800px，兼顾手机性能和精度
      const MAX_WIDTH = 800;
      let width = img.width;
      let height = img.height;

      if (width > MAX_WIDTH) {
        height = Math.round((height * MAX_WIDTH) / width);
        width = MAX_WIDTH;
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);

      document.getElementById('resultSection').classList.remove('hidden');
      
      if (cvReady) {
        processImage();
      } else {
        alert("算法引擎正在加载中，请稍候...");
      }
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function processImage() {
  if (!cvReady) return;

  // 如果没有缓存矩阵，则从 canvas 读取
  if (srcImage) srcImage.delete();
  srcImage = cv.imread('originalCanvas');

  // 获取 UI 参数
  let blurSize = parseInt(document.getElementById('blurSize').value);
  blurSize = blurSize % 2 === 0 ? blurSize + 1 : blurSize;
  let minArea = parseInt(document.getElementById('minArea').value);
  let maxArea = parseInt(document.getElementById('maxArea').value);
  let kernelSize = parseInt(document.getElementById('kernelSize').value);
  kernelSize = kernelSize % 2 === 0 ? kernelSize + 1 : kernelSize;
  let separationLevel = parseInt(document.getElementById('separationLevel').value) / 100.0;
  let minCircularity = parseInt(document.getElementById('minCircularity').value) / 100.0;
  let invert = document.getElementById('invertToggle').checked;

  let gray = new cv.Mat();
  let blur = new cv.Mat();
  let thresh = new cv.Mat();
  let distTrans = new cv.Mat();
  let sureFg = new cv.Mat();
  let hierarchy = new cv.Mat();
  let contours = new cv.MatVector();
  let result = srcImage.clone();

  try {
    cv.cvtColor(srcImage, gray, cv.COLOR_RGBA2GRAY, 0);
    cv.GaussianBlur(gray, blur, new cv.Size(blurSize, blurSize), 0, 0, cv.BORDER_DEFAULT);

    // 自适应阈值对抗阴影
    if (invert) {
      cv.adaptiveThreshold(blur, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10);
    } else {
      cv.adaptiveThreshold(blur, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10);
    }

    let M = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kernelSize, kernelSize));
    cv.morphologyEx(thresh, thresh, cv.MORPH_OPEN, M);

    // 核心：距离变换分离粘连
    cv.distanceTransform(thresh, distTrans, cv.DIST_L2, 5);
    cv.normalize(distTrans, distTrans, 1, 0, cv.NORM_INF);
    cv.threshold(distTrans, sureFg, separationLevel, 255, cv.THRESH_BINARY);
    sureFg.convertTo(sureFg, cv.CV_8U);

    cv.findContours(sureFg, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let count = 0;
    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);
      let area = cv.contourArea(cnt);
      let perimeter = cv.arcLength(cnt, true);
      let circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;

      // 距离变换后的核心面积较小，故系数下调
      if (area >= minArea * 0.2 && area <= maxArea && circularity >= minCircularity) {
        count++;
        let moments = cv.moments(cnt, false);
        let cx = moments.m10 / moments.m00;
        let cy = moments.m01 / moments.m00;
        cv.drawContours(result, contours, i, new cv.Scalar(0, 255, 0, 255), 2);
        cv.putText(result, count.toString(), new cv.Point(cx - 5, cy + 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, new cv.Scalar(255, 0, 0, 255), 1);
      }
      cnt.delete();
    }

    document.getElementById('seedCount').innerText = count;
    cv.imshow('resultCanvas', result);
    cv.imshow('threshCanvas', sureFg);

  } catch (err) {
    console.error(err);
  } finally {
    gray.delete(); blur.delete(); thresh.delete(); distTrans.delete();
    sureFg.delete(); hierarchy.delete(); contours.delete(); result.delete();
    if (typeof M !== 'undefined') M.delete();
  }
}
