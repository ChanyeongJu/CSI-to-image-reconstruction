#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "driver/uart.h"
#include "driver/gpio.h"

#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "esp_now.h"

#define ESP_NOW_CHANNEL 11 

#define UART_BAUD_RATE  921600
#define UART_PORT_NUM   UART_NUM_1
#define TXD_PIN         GPIO_NUM_1
#define RXD_PIN         GPIO_NUM_2
#define BUF_SIZE        2048

#define TAG             "CSI-RX"

static QueueHandle_t s_csi_queue = NULL;

static const char *const CSI_PREFIX_FORMAT = "\"" MACSTR "\",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,\"[%d";
static const char *const CSI_SUFFIX_FORMAT = "]\"\n";
static const char *const CSI_BUF_ELEMENT_FORMAT = ",%d";

void serial_sender_task(void *pvParameter) {
    wifi_csi_info_t info;
    wifi_pkt_rx_ctrl_t *rx_ctrl;

    char *data_to_send = malloc(BUF_SIZE);
    if (!data_to_send) {
        ESP_LOGE(TAG, "Failed to allocate memory for UART send buffer. Task aborting.");
        vTaskDelete(NULL);
        return;
    }
    
    while(1) {
        if (xQueueReceive(s_csi_queue, &info, portMAX_DELAY) == pdPASS) {
            rx_ctrl = &info.rx_ctrl;
            int len = 0;

            len = snprintf(data_to_send, BUF_SIZE,
                CSI_PREFIX_FORMAT,
                MAC2STR(info.mac),
                rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode, rx_ctrl->mcs,
                rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
                rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding,
                rx_ctrl->sgi, rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt,
                rx_ctrl->channel, rx_ctrl->secondary_channel, rx_ctrl->timestamp,
                rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->rx_state,
                info.len, info.first_word_invalid, info.buf[0]);

            for (int i = 1; i < info.len; i++) {
                len += snprintf(data_to_send + len, BUF_SIZE - len,
                                CSI_BUF_ELEMENT_FORMAT, info.buf[i]);
            }

            len += snprintf(data_to_send + len, BUF_SIZE - len, CSI_SUFFIX_FORMAT);
            
            uart_write_bytes(UART_PORT_NUM, data_to_send, len);

            free(info.buf);
        }
    }
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "Receive csi info error");
        return;
    }

    wifi_csi_info_t data_to_queue;
    memcpy(&data_to_queue, info, sizeof(wifi_csi_info_t));
    
    data_to_queue.buf = malloc(info->len);
    if (data_to_queue.buf == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for CSI buffer");
        return;
    }
    memcpy(data_to_queue.buf, info->buf, info->len);

    if (xQueueSend(s_csi_queue, &data_to_queue, 0) != pdPASS) {
        ESP_LOGW(TAG, "CSI queue is full, dropping data.");
        free(data_to_queue.buf);
    }
}

static void csi_init(void)
{
    const wifi_csi_config_t csi_config = {
        .lltf_en           = false,
        .htltf_en          = true,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = true,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    
    ESP_LOGI(TAG, "CSI initialized successfully.");
}

static void wifi_init(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_channel(ESP_NOW_CHANNEL, WIFI_SECOND_CHAN_BELOW));

    ESP_LOGI(TAG, "Wi-Fi initialized successfully on Channel %d (40MHz Below)", ESP_NOW_CHANNEL);
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

static void print_mac_address(void)
{
    uint8_t mac_addr[6] = {0};
    esp_wifi_get_mac(WIFI_IF_STA, mac_addr);
    ESP_LOGI(TAG, "===================================================================");
    ESP_LOGI(TAG, "RX Board MAC Address: " MACSTR, MAC2STR(mac_addr));
    ESP_LOGI(TAG, "===================================================================");
}

void app_main(void) {
    wifi_init();

    print_mac_address();

    ESP_ERROR_CHECK(esp_now_init());

    s_csi_queue = xQueueCreate(16, sizeof(wifi_csi_info_t));
    if (s_csi_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create CSI queue");
        return;
    }

    csi_init();

    uart_init();
    
    xTaskCreate(&serial_sender_task, "serial_sender_task", 4096, NULL, 5, NULL);
}
