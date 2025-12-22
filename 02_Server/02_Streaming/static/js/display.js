const imageElement = document.getElementById('image')
const statusElement = document.getElementById('status-message')

const WEBSOCKET_URL = `ws://${window.location.host}/ws`

const socket = new WebSocket(WEBSOCKET_URL)

socket.onopen = (event) => {
    console.log("WebSocket connection established successfully!")
    statusElement.textContent = "Connection successful! Receiving image..."
}

socket.onmessage = (event) => {
    if (imageElement.style.display === 'none') {
        statusElement.style.display = 'none'
        imageElement.style.display = 'block'
    }

    image = JSON.parse(event.data)
    imageElement.src = "data:image/jpeg;base64," + image.image
}

socket.onerror = (error) => {
    console.error("WebSocket error occurred:", error)
    statusElement.textContent = "A connection error occurred. Please check the server status."
    statusElement.style.color = "#ff6b6b"
}

socket.onclose = (event) => {
    console.log("WebSocket connection closed.")
    statusElement.textContent = "The connection to the server has been lost. Please refresh the page."
    statusElement.style.display = 'block'
    imageElement.style.display = 'none'
}