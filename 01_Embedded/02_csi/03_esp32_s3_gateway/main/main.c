#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "driver/uart.h"
#include "driver/gpio.h"

#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"

#include "lwip/err.h"
#include "lwip/sockets.h"

#include "protocol_examples_common.h"

#define UART_BAUD_RATE  921600
#define UART_PORT_NUM   UART_NUM_1
#define TXD_PIN         GPIO_NUM_1
#define RXD_PIN         GPIO_NUM_2
#define BUF_SIZE        2048

#define UDP_SERVER_IP   "192.168.7.45"
#define UDP_SERVER_PORT 8000

#define TAG             "CSI-GATEWAY"

static void udp_csi_send_task(void *pvParameters)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "UDP socket created, sending to %s:%d", UDP_SERVER_IP, UDP_SERVER_PORT);

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(UDP_SERVER_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(UDP_SERVER_PORT);

    static char uart_buffer[BUF_SIZE];
    static int buffer_len = 0;

    while (1) {
        int len = uart_read_bytes(UART_PORT_NUM, uart_buffer + buffer_len, BUF_SIZE - buffer_len, 20 / portTICK_PERIOD_MS);

        if (len > 0) {
            buffer_len += len;

            char *newline_ptr;
            while ((newline_ptr = memchr(uart_buffer, '\n', buffer_len)) != NULL) {
                int packet_len = (newline_ptr - uart_buffer) + 1;

                sendto(sock, uart_buffer, packet_len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));

                buffer_len -= packet_len;
                if (buffer_len > 0) {
                    memmove(uart_buffer, uart_buffer + packet_len, buffer_len);
                }
            }
        }

        if (buffer_len == BUF_SIZE) {
            buffer_len = 0;
        }
    }
}

void uart_init() {
    uart_config_t uart_config = {
        .baud_rate  = UART_BAUD_RATE,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_driver_install(UART_PORT_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);
    uart_param_config(UART_PORT_NUM, &uart_config);
    uart_set_pin(UART_PORT_NUM, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
}

void app_main(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    uart_init();

    ESP_ERROR_CHECK(example_connect());

    xTaskCreate(&udp_csi_send_task, "udp_csi_send_task", 4096, NULL, 10, NULL);
}