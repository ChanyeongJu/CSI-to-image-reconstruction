#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "nvs_flash.h"

#include "esp_log.h"
#include "esp_camera.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"

#include "lwip/sockets.h"

#include "protocol_examples_common.h"

#define UDP_SERVER_IP   "192.168.7.45"
#define UDP_SERVER_PORT 8001

#define CONFIG_SEND_FREQUENCY 20

static const char *TAG = "ESP32_S3_CAM";

static camera_config_t camera_config = {
    .pin_pwdn       = -1,
    .pin_reset      = -1,
    .pin_xclk       = 15,
    .pin_sccb_sda   = 4,
    .pin_sccb_scl   = 5,
    .pin_d7         = 16,
    .pin_d6         = 17,
    .pin_d5         = 18,
    .pin_d4         = 12,
    .pin_d3         = 10,
    .pin_d2         = 8,
    .pin_d1         = 9,
    .pin_d0         = 11,
    .pin_vsync      = 6,
    .pin_href       = 7,
    .pin_pclk       = 13,

    .xclk_freq_hz   = 20000000,
    .pixel_format   = PIXFORMAT_JPEG,
    .frame_size     = FRAMESIZE_QVGA,
    .jpeg_quality   = 30,
    .fb_count       = 1,

    .fb_location    = CAMERA_FB_IN_DRAM,
};

static esp_err_t init_camera(void)
{
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed");
        return err;
    }
    return err;
}

void udp_image_send_task(void *pvParameters) {
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket, errno: %d", errno);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "UDP socket created, sending to %s:%d", UDP_SERVER_IP, UDP_SERVER_PORT);

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(UDP_SERVER_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(UDP_SERVER_PORT);

    while (1) {
        camera_fb_t *pic = esp_camera_fb_get();
        if (!pic) {
            ESP_LOGE(TAG, "Failed to capture image");
            vTaskDelay(pdMS_TO_TICKS(1000 / CONFIG_SEND_FREQUENCY));
            continue;
        }

        sendto(sock, pic->buf, pic->len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));

        esp_camera_fb_return(pic);
        
        vTaskDelay(pdMS_TO_TICKS(1000 / CONFIG_SEND_FREQUENCY)); 
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(example_connect());

    if (ESP_OK != init_camera()) {
        return;
    }

    xTaskCreate(udp_image_send_task, "udp_image_send_task", 8192, NULL, 5, NULL);
}
