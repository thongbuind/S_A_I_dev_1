/* ===== THEME TOGGLE ===== */
const html = document.documentElement;
const themeToggle = document.getElementById('themeToggle');

themeToggle.addEventListener('click', () => {
  window.SAITheme.toggle();
});

/* ===== POINT CLOUD ===== */
const canvas = document.getElementById("bg-canvas");
const ctx = canvas.getContext("2d");
let W, H, points;

const PARTICLE_COLOR_TOKENS = [
  "--color-particle-1",
  "--color-particle-2",
  "--color-particle-3",
  "--color-particle-4",
  "--color-particle-5",
  "--color-particle-6",
];
const ALPHA_RANGE = [0.12, 0.5];
let COLORS = readParticleColors();
const SPEED = 1.2;
const POINT_COUNT = 480;
const MOUSE = { x: -9999, y: -9999 };

function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
function readParticleColors() {
  const style = getComputedStyle(html);
  return PARTICLE_COLOR_TOKENS
    .map(token => style.getPropertyValue(token).trim().replace(/\s+/g, ","))
    .filter(Boolean);
}
function initPoints() {
  points = Array.from({ length: POINT_COUNT }, () => ({
    x: Math.random()*W, y: Math.random()*H,
    vx: (Math.random()-0.5)*SPEED, vy: (Math.random()-0.5)*SPEED,
    r: Math.random()*2+0.4,
    color: COLORS[Math.floor(Math.random()*COLORS.length)],
    alpha: Math.random()*(ALPHA_RANGE[1]-ALPHA_RANGE[0])+ALPHA_RANGE[0],
    wx: Math.random()*Math.PI*2, wy: Math.random()*Math.PI*2,
    wsx: (Math.random()*0.008+0.003)*(Math.random()<0.5?1:-1),
    wsy: (Math.random()*0.008+0.003)*(Math.random()<0.5?1:-1),
    wax: Math.random()*0.6+0.2, way: Math.random()*0.6+0.2,
  }));
}
function updatePointColors() {
  COLORS = readParticleColors();
  // Smoothly update existing points
  if (points) {
    points.forEach(p => {
      p.color = COLORS[Math.floor(Math.random()*COLORS.length)];
      p.alpha = Math.random()*(ALPHA_RANGE[1]-ALPHA_RANGE[0])+ALPHA_RANGE[0];
    });
  }
}
window.addEventListener('sai-theme-change', updatePointColors);
function draw() {
  ctx.clearRect(0,0,W,H);
  for (const p of points) {
    p.wx+=p.wsx; p.wy+=p.wsy;
    p.x+=p.vx+Math.sin(p.wx)*p.wax; p.y+=p.vy+Math.cos(p.wy)*p.way;
    if (p.x<-10) p.x=W+10; if (p.x>W+10) p.x=-10;
    if (p.y<-10) p.y=H+10; if (p.y>H+10) p.y=-10;
    const dx=p.x-MOUSE.x, dy=p.y-MOUSE.y, dist=Math.hypot(dx,dy);
    if (dist<120&&dist>0){const f=(120-dist)/120; p.x+=(dx/dist)*f*1.4; p.y+=(dy/dist)*f*1.4;}
    ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx.fillStyle=`rgba(${p.color},${p.alpha})`; ctx.fill();
  }
  requestAnimationFrame(draw);
}
window.addEventListener("mousemove",e=>{MOUSE.x=e.clientX;MOUSE.y=e.clientY;});
window.addEventListener("mouseleave",()=>{MOUSE.x=-9999;MOUSE.y=-9999;});
window.addEventListener("resize",()=>{resize();initPoints();});
resize(); initPoints(); draw();

/* ===== NAVBAR SCROLL ===== */
window.addEventListener("scroll",()=>{
  const nb = document.getElementById("navbar");
  if (window.scrollY > 60) {
    nb.style.boxShadow = "var(--navbar-shadow-scroll)";
  } else {
    nb.style.boxShadow = "";
  }
});

/* ===== SCROLL REVEAL ===== */
const io = new IntersectionObserver((entries)=>{
  entries.forEach(e=>{ if(e.isIntersecting){e.target.classList.add("in-view");io.unobserve(e.target);} });
},{threshold:0.12});
document.querySelectorAll(".reveal").forEach(el=>io.observe(el));

/* ===== ARCHITECTURE TABLE COLUMN HOVER ===== */
document.querySelectorAll(".architecture-table").forEach(table => {
  table.addEventListener("mouseover", event => {
    const cell = event.target.closest(".architecture-table > div > div");
    if (!cell || !table.contains(cell)) return;
    const column = Array.from(cell.parentElement.children).indexOf(cell) + 1;
    if (column > 1) table.dataset.hoverColumn = column;
    else delete table.dataset.hoverColumn;
  });
  table.addEventListener("mouseleave", () => delete table.dataset.hoverColumn);
});

/* ===== CHAT TYPEWRITER ===== */
const AI_TEXT = "Transformer là kiến trúc nền tảng của các mô hình AI hiện đại, dựa trên cơ chế \"attention\" cho phép mô hình hiểu mối quan hệ giữa các từ trong toàn bộ câu cùng một lúc.";

let chatTriggered = false;
const chatObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting && !chatTriggered) {
      chatTriggered = true;
      chatObs.unobserve(e.target);
      startChatAnimation();
    }
  });
}, { threshold: 0.5 });
chatObs.observe(document.getElementById("chat-visual"));

function startChatAnimation() {
  const typing = document.getElementById("cp-typing");
  const bubble = document.getElementById("cp-ai-bubble");
  setTimeout(() => {
    typing.style.transition = "opacity 0.2s ease";
    typing.style.opacity = "0";
    setTimeout(() => {
      typing.style.display = "none";
      bubble.style.display = "block";
      bubble.textContent = "";
      let i = 0;
      function type() {
        if (i < AI_TEXT.length) { bubble.textContent += AI_TEXT[i++]; setTimeout(type, 18); }
      }
      type();
    }, 220);
  }, 2000);
}
