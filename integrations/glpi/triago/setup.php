<?php
/**
 * triago for GLPI: route a ticket while it is being created, so it opens already assigned.
 *
 * On ticket creation GLPI calls plugin_triago_pre_item_add() before saving; we ask the local
 * triago server (python -m triago.serve) and add the team, impact, urgency and priority to the
 * ticket being saved. After saving, plugin_triago_item_add() writes triago's private note.
 * If triago is slow or down, the ticket is saved unrouted and `python -m triago.dispatcher run`
 * routes it a few seconds later: this plugin is the fast path, the dispatcher the safety net.
 *
 * Settings (environment of the GLPI container, all optional):
 *   TRIAGO_URL       default http://host.docker.internal:8642
 *   TRIAGO_API_KEY   if the triago server was started with one
 *   TRIAGO_TIMEOUT   seconds, default 4
 */

use Glpi\Plugin\Hooks;

define('PLUGIN_TRIAGO_VERSION', '0.1.0');

function plugin_version_triago()
{
    return [
        'name'         => 'triago',
        'version'      => PLUGIN_TRIAGO_VERSION,
        'author'       => 'triago contributors',
        'license'      => 'Apache-2.0',
        'homepage'     => 'https://github.com/eduardd76/triago',
        'requirements' => ['glpi' => ['min' => '10.0.0']],
    ];
}

function plugin_init_triago()
{
    global $PLUGIN_HOOKS;
    $PLUGIN_HOOKS['csrf_compliant']['triago'] = true;
    $PLUGIN_HOOKS[Hooks::PRE_ITEM_ADD]['triago'] = ['Ticket' => 'plugin_triago_pre_item_add'];
    $PLUGIN_HOOKS[Hooks::ITEM_ADD]['triago']     = ['Ticket' => 'plugin_triago_item_add'];
}

function plugin_triago_check_prerequisites()
{
    return function_exists('curl_init');
}

function plugin_triago_check_config($verbose = false)
{
    return true;
}
