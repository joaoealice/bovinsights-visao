// Service Worker mínimo — apenas registra para habilitar o PWA
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", () => self.clients.claim());
// Cache offline pode ser adicionado futuramente
