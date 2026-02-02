#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "nvs_flash.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_now.h"

#define ESP_NOW_CHANNEL         11
#define CONFIG_SEND_FREQUENCY   100

static const char *TAG = "CSI-TX";

static uint8_t s_peer_mac[ESP_NOW_ETH_ALEN] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static void esp_now_send_cb(const uint8_t *mac_addr, esp_now_send_status_t status)
{
    if (mac_addr == NULL) {
        ESP_LOGE(TAG, "Send CB error: MAC address is NULL");
        return;
    }
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
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(84));

    ESP_ERROR_CHECK(esp_wifi_set_channel(ESP_NOW_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    
    ESP_LOGI(TAG, "Wi-Fi initialized successfully on Channel %d (40MHz Below)", ESP_NOW_CHANNEL);
}

static void my_esp_now_init(void)
{
    ESP_ERROR_CHECK(esp_now_init());

    ESP_ERROR_CHECK(esp_now_register_send_cb(esp_now_send_cb));

    esp_now_peer_info_t peer = {0};
    peer.channel = ESP_NOW_CHANNEL;
    peer.ifidx = ESP_IF_WIFI_STA;
    peer.encrypt = false;
    memcpy(peer.peer_addr, s_peer_mac, ESP_NOW_ETH_ALEN);

    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    esp_now_rate_config_t rate_config = {
        .phymode = WIFI_PHY_MODE_HT40,
        .rate    = WIFI_PHY_RATE_MCS7_SGI,
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));

    ESP_LOGI(TAG, "ESP-NOW initialized and peer rate configured for CSI.");
}

void tx_task(void *pvParameter)
{
    uint8_t empty_data = 0;

    while (1) {
        esp_now_send(s_peer_mac, &empty_data, sizeof(empty_data));
        
        vTaskDelay(pdMS_TO_TICKS(1000 / CONFIG_SEND_FREQUENCY));
    }
}

void app_main(void)
{
    wifi_init();
    
    my_esp_now_init();

    xTaskCreate(tx_task, "tx_task", 4096, NULL, 5, NULL);
}