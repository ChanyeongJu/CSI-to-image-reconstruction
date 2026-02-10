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
#include "nvs.h"

#include "lwip/err.h"
#include "lwip/sockets.h"

#define UART_BAUD_RATE  921600
#define UART_PORT_NUM   UART_NUM_1
#define TXD_PIN         GPIO_NUM_1
#define RXD_PIN         GPIO_NUM_2
#define BUF_SIZE        2048

#define DEFAULT_WIFI_SSID   "WIFI_SSID"
#define DEFAULT_WIFI_PWD    "WIFI_PASSWORD"
#define DEFAULT_SERVER_IP   "192.168.1.1"
#define DEFAULT_SERVER_PORT 8000

static char s_wifi_ssid[32] = DEFAULT_WIFI_SSID;
static char s_wifi_pwd[64] = DEFAULT_WIFI_PWD;
static char s_server_ip[32] = DEFAULT_SERVER_IP;
static int s_server_port = DEFAULT_SERVER_PORT;

#define TAG             "CSI-GATEWAY"

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        ESP_LOGI(TAG, "retry to connect to the AP");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_init_sta(void)
{
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = {
                .capable = true,
                .required = false
            },
        },
    };
    strncpy((char*)wifi_config.sta.ssid, s_wifi_ssid, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, s_wifi_pwd, sizeof(wifi_config.sta.password));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished. SSID:%s", s_wifi_ssid);
}

static void udp_csi_send_task(void *pvParameters)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "UDP socket created, sending to %s:%d", s_server_ip, s_server_port);

    struct sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(s_server_port);

    static char uart_buffer[BUF_SIZE];
    static int buffer_len = 0;

    while (1) {
        int len = uart_read_bytes(UART_PORT_NUM, uart_buffer + buffer_len, BUF_SIZE - buffer_len, 20 / portTICK_PERIOD_MS);

        if (len > 0) {
            buffer_len += len;

            char *newline_ptr;
            while ((newline_ptr = memchr(uart_buffer, '\n', buffer_len)) != NULL) {
                int packet_len = (newline_ptr - uart_buffer) + 1;

                dest_addr.sin_addr.s_addr = inet_addr(s_server_ip);
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

static void nvs_load_config(void) {
    nvs_handle_t handle;
    if (nvs_open("storage", NVS_READONLY, &handle) == ESP_OK) {
        size_t size = sizeof(s_wifi_ssid);
        nvs_get_str(handle, "wifi_ssid", s_wifi_ssid, &size);
        size = sizeof(s_wifi_pwd);
        nvs_get_str(handle, "wifi_pwd", s_wifi_pwd, &size);
        size = sizeof(s_server_ip);
        nvs_get_str(handle, "server_ip", s_server_ip, &size);
        nvs_get_i32(handle, "server_port", &s_server_port);
        nvs_close(handle);
        ESP_LOGI(TAG, "Loaded config from NVS: %s:%d (SSID: %s)", s_server_ip, s_server_port, s_wifi_ssid);
    }
}

static void nvs_save_config(void) {
    nvs_handle_t handle;
    if (nvs_open("storage", NVS_READWRITE, &handle) == ESP_OK) {
        nvs_set_str(handle, "wifi_ssid", s_wifi_ssid);
        nvs_set_str(handle, "wifi_pwd", s_wifi_pwd);
        nvs_set_str(handle, "server_ip", s_server_ip);
        nvs_set_i32(handle, "server_port", s_server_port);
        nvs_commit(handle);
        nvs_close(handle);
        ESP_LOGI(TAG, "Saved config to NVS");
    }
}

static void process_command(char *line) {
    if (strlen(line) == 0) return;

    if (strncmp(line, "SET_SSID:", 9) == 0) {
        char *ssid = line + 9;
        strncpy(s_wifi_ssid, ssid, sizeof(s_wifi_ssid) - 1);
        s_wifi_ssid[sizeof(s_wifi_ssid) - 1] = '\0';
        nvs_save_config();
        char msg[128];
        snprintf(msg, sizeof(msg), "[OK] SSID:%s\n", s_wifi_ssid);
        uart_write_bytes(UART_NUM_0, msg, strlen(msg));
    } else if (strncmp(line, "SET_PWD:", 8) == 0) {
        char *pwd = line + 8;
        strncpy(s_wifi_pwd, pwd, sizeof(s_wifi_pwd) - 1);
        s_wifi_pwd[sizeof(s_wifi_pwd) - 1] = '\0';
        nvs_save_config();
        char msg[128];
        snprintf(msg, sizeof(msg), "[OK] PWD:%s\n", s_wifi_pwd);
        uart_write_bytes(UART_NUM_0, msg, strlen(msg));
    } else if (strncmp(line, "SET_IP:", 7) == 0) {
        char *ip = line + 7;
        strncpy(s_server_ip, ip, sizeof(s_server_ip) - 1);
        s_server_ip[sizeof(s_server_ip) - 1] = '\0';
        nvs_save_config();
        char msg[128];
        snprintf(msg, sizeof(msg), "[OK] IP:%s\n", s_server_ip);
        uart_write_bytes(UART_NUM_0, msg, strlen(msg));
    } else if (strncmp(line, "SET_PORT:", 9) == 0) {
        char *port_str = line + 9;
        s_server_port = atoi(port_str);
        nvs_save_config();
        char msg[128];
        snprintf(msg, sizeof(msg), "[OK] PORT:%d\n", s_server_port);
        uart_write_bytes(UART_NUM_0, msg, strlen(msg));
    } else if (strncmp(line, "RESTART", 7) == 0) {
        uart_write_bytes(UART_NUM_0, "[SYSTEM] Restarting...\n", 23);
        vTaskDelay(pdMS_TO_TICKS(500));
        esp_restart();
    } else if (strncmp(line, "GET_CONFIG", 10) == 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "[INFO] Current Config - SSID:%s, PWD:%s, IP:%s, Port:%d\n", 
                 s_wifi_ssid, s_wifi_pwd, s_server_ip, s_server_port);
        uart_write_bytes(UART_NUM_0, msg, strlen(msg));
    } else if (strncmp(line, "HELP", 4) == 0) {
        const char* help = "\n--- Commands ---\nSET_SSID:xxxx\nSET_PWD:xxxx\nSET_IP:x.x.x.x\nSET_PORT:xxxx\nGET_CONFIG\nRESTART\n-----------------\n";
        uart_write_bytes(UART_NUM_0, help, strlen(help));
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "[ERR] Unknown command: %s\n", line);
        uart_write_bytes(UART_NUM_0, msg, strlen(msg));
    }
}

void uart_console_task(void *pvParameters) {
    uint8_t *buf = (uint8_t *) malloc(1024);
    char line[128];
    int line_idx = 0;
    
    uart_write_bytes(UART_NUM_0, "\n[SYSTEM] ESP32-S3 Gateway Ready. Type HELP for commands.\n", 58);
    
    while (1) {
        int len = uart_read_bytes(UART_NUM_0, buf, 1024, 20 / portTICK_PERIOD_MS);
        for (int i = 0; i < len; i++) {
            if (buf[i] == '\n' || buf[i] == '\r') {
                if (line_idx > 0) {
                    line[line_idx] = '\0';
                    process_command(line);
                    line_idx = 0;
                }
            } else {
                if (line_idx < sizeof(line) - 1) {
                    line[line_idx++] = buf[i];
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    free(buf);
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
    nvs_load_config();

    wifi_init_sta();

    uart_config_t uart0_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_0, &uart0_config);

    xTaskCreate(udp_csi_send_task, "udp_csi_send_task", 4096, NULL, 10, NULL);
    xTaskCreate(uart_console_task, "uart_console_task", 4096, NULL, 1, NULL);
}