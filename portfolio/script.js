// Generate random geometric background elements
function createGeometricBackground() {
    const geometricBg = document.getElementById('geometricBg');
    
    // Clear existing elements first
    geometricBg.innerHTML = '';
    
    const elementCount = 50;
    
    for (let i = 0; i < elementCount; i++) {
        const element = document.createElement('div');
        element.classList.add('geometric-element');
        
        const elementType = Math.random();
        
        if (elementType < 0.6) {
            // Create squares (60% chance)
            element.classList.add('geometric-square');
            const sizeRandom = Math.random();
            if (sizeRandom < 0.3) {
                element.classList.add('small');
            } else if (sizeRandom > 0.7) {
                element.classList.add('large');
            }
        } else if (elementType < 0.85) {
            // Create lines (25% chance)
            element.classList.add('geometric-line');
            const lengthRandom = Math.random();
            if (lengthRandom < 0.33) {
                element.classList.add('short');
            } else if (lengthRandom < 0.66) {
                element.classList.add('medium');
            } else {
                element.classList.add('long');
            }
            element.style.transform = `rotate(${Math.random() * 360}deg)`;
        } else {
            // Create triangles (15% chance)
            element.classList.add('geometric-triangle');
        }
        
        // Random position
        element.style.left = Math.random() * 100 + '%';
        element.style.top = Math.random() * 100 + '%';
        
        // Random animation delay
        element.style.animationDelay = Math.random() * 10 + 's';
        
        geometricBg.appendChild(element);
    }
}

// Ensure this only runs once
let isInitialized = false;

document.addEventListener('DOMContentLoaded', function() {
    if (!isInitialized) {
        createGeometricBackground();
        isInitialized = true;
    }
    
    const projectCards = document.querySelectorAll('.project-card');
    
    projectCards.forEach((card, index) => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
            this.style.boxShadow = '0 15px 35px rgba(0, 0, 0, 0.1)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.5)';
        });
    });
});

// Add paper aging spots dynamically
function addPaperAging() {
    const container = document.querySelector('.container');
    
    // Create aging spots
    for (let i = 0; i < 8; i++) {
        const spot = document.createElement('div');
        spot.style.position = 'absolute';
        spot.style.width = `${10 + Math.random() * 20}px`;
        spot.style.height = `${10 + Math.random() * 20}px`;
        spot.style.background = `radial-gradient(circle, rgba(120, 119, 108, ${0.1 + Math.random() * 0.1}) 0%, transparent 70%)`;
        spot.style.borderRadius = '50%';
        spot.style.top = `${Math.random() * 100}%`;
        spot.style.left = `${Math.random() * 100}%`;
        spot.style.pointerEvents = 'none';
        spot.style.zIndex = '2';
        
        container.appendChild(spot);
    }
}

// Update geometric background for paper effect
function createPaperGeometricBackground() {
    const geometricBg = document.getElementById('geometricBg');
    const elementCount = 25;
    
    for (let i = 0; i < elementCount; i++) {
        const element = document.createElement('div');
        element.style.position = 'absolute';
        element.style.opacity = '0.1';
        
        const elementType = Math.random();
        
        if (elementType < 0.6) {
            element.style.width = '6px';
            element.style.height = '6px';
            element.style.background = '#1a1a1a';
            element.style.transform = 'rotate(45deg)';
            element.style.borderRadius = '1px';
        } else {
            element.style.width = '30px';
            element.style.height = '1px';
            element.style.background = '#1a1a1a';
            element.style.transform = `rotate(${Math.random() * 360}deg)`;
        }
        
        element.style.left = Math.random() * 100 + '%';
        element.style.top = Math.random() * 100 + '%';
        element.style.animation = `float ${12 + Math.random() * 6}s ease-in-out infinite`;
        element.style.animationDelay = Math.random() * 10 + 's';
        
        geometricBg.appendChild(element);
    }
}

// Initialize paper effects
document.addEventListener('DOMContentLoaded', () => {
    addPaperAging();
    createPaperGeometricBackground();
    new VerticalBreadcrumb();
});
