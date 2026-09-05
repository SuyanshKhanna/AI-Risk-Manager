class AmbientParticles {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    this.particles = [];
    this.mouse = { x: -1000, y: -1000 };
    /* Using the accent-default color (#1e2a4a) with low opacity */
    this.colors = [
      'rgba(30, 42, 74, 0.08)', 
      'rgba(30, 42, 74, 0.12)', 
      'rgba(30, 42, 74, 0.05)'
    ];
    
    this.init();
    this.animate();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
    window.addEventListener('mouseout', () => {
      this.mouse.x = -1000;
      this.mouse.y = -1000;
    });

    const numParticles = Math.floor((this.width * this.height) / 12000);
    
    for (let i = 0; i < numParticles; i++) {
      this.particles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        baseVx: (Math.random() - 0.5) * 0.3,
        baseVy: (Math.random() - 0.5) * 0.3,
        radius: Math.random() * 1.5 + 0.5,
        color: this.colors[Math.floor(Math.random() * this.colors.length)]
      });
    }
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.ctx.scale(dpr, dpr);
  }

  animate() {
    this.ctx.clearRect(0, 0, this.width, this.height);
    
    // Update and draw particles
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      
      // Mouse interaction (gentle repulsion)
      const dx = this.mouse.x - p.x;
      const dy = this.mouse.y - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < 150) {
        const force = (150 - dist) / 150;
        p.vx -= (dx / dist) * force * 0.03;
        p.vy -= (dy / dist) * force * 0.03;
      }
      
      // Return gently to base velocity
      p.vx += (p.baseVx - p.vx) * 0.02;
      p.vy += (p.baseVy - p.vy) * 0.02;

      p.x += p.vx;
      p.y += p.vy;

      // Wrap around
      if (p.x < 0) p.x = this.width;
      if (p.x > this.width) p.x = 0;
      if (p.y < 0) p.y = this.height;
      if (p.y > this.height) p.y = 0;

      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = p.color;
      this.ctx.fill();

      // Connect nearby particles
      for (let j = i + 1; j < this.particles.length; j++) {
        const p2 = this.particles[j];
        const dx2 = p2.x - p.x;
        const dy2 = p2.y - p.y;
        const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);

        if (dist2 < 120) {
          this.ctx.beginPath();
          this.ctx.moveTo(p.x, p.y);
          this.ctx.lineTo(p2.x, p2.y);
          // Very faint lines linking close particles
          this.ctx.strokeStyle = `rgba(30, 42, 74, ${0.04 * (1 - dist2 / 120)})`;
          this.ctx.lineWidth = 0.5;
          this.ctx.stroke();
        }
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new AmbientParticles('ambient-canvas');
});
