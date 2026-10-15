<?php
/** triago for GLPI: the two ticket hooks. See setup.php for what this does and how to configure it. */

// triago's group keys -> GLPI group names (same as triago/glpi.py GROUP_NAMES).
const PLUGIN_TRIAGO_GROUPS = [
    'security'        => 'Security Operations (SOC)',
    'network'         => 'Network',
    'server_infra'    => 'Server & Infrastructure',
    'end_user'        => 'End-user Computing',
    'identity_access' => 'Identity & Access',
    'application'     => 'Business Applications',
    'telephony'       => 'Telephony',
    'facilities'      => 'Facilities',
    'triage'          => 'Service Desk Triage',
];
const PLUGIN_TRIAGO_LEVEL    = ['high' => 4, 'medium' => 3, 'low' => 2];        // GLPI 1..5 scale
const PLUGIN_TRIAGO_PRIORITY = [1 => 6, 2 => 5, 3 => 4, 4 => 3, 5 => 2];        // P1..P5 -> GLPI 6..2

function plugin_triago_install()
{
    return true;
}

function plugin_triago_uninstall()
{
    return true;
}

/** POST {subject, body} to triago; returns the decoded decision or null (never throws). */
function plugin_triago_score(string $subject, string $body): ?array
{
    $url = rtrim(getenv('TRIAGO_URL') ?: 'http://host.docker.internal:8642', '/') . '/dispatch';
    $headers = ['Content-Type: application/json'];
    if ($key = getenv('TRIAGO_API_KEY')) {
        $headers[] = 'Authorization: Bearer ' . $key;
    }
    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_POST           => true,
        CURLOPT_POSTFIELDS     => json_encode(['subject' => $subject, 'body' => $body]),
        CURLOPT_HTTPHEADER     => $headers,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_CONNECTTIMEOUT => 1,
        CURLOPT_TIMEOUT        => (int) (getenv('TRIAGO_TIMEOUT') ?: 4),
    ]);
    $raw  = curl_exec($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err  = curl_error($ch);
    curl_close($ch);
    if ($raw === false || $code !== 200) {
        Toolbox::logInFile('triago', sprintf("triago not used (HTTP %s %s); the dispatcher will route it\n", $code, $err));
        return null;
    }
    $d = json_decode($raw, true);
    return is_array($d) && isset($d['assign_to']) ? $d : null;
}

function plugin_triago_group_id(string $name): ?int
{
    $group = new Group();
    return $group->getFromDBByCrit(['name' => $name]) ? (int) $group->getID() : null;
}

/** Before the ticket is saved: add team, impact, urgency, priority to what is being saved. */
function plugin_triago_pre_item_add(Ticket $ticket)
{
    $input = &$ticket->input;
    if (!is_array($input) || !empty($input['_triago_skip'])) {
        return;
    }
    $subject = (string) ($input['name'] ?? '');
    // content is HTML, possibly entity-encoded (older GLPI sanitizes input): decode, drop tags, decode text
    $html    = html_entity_decode((string) ($input['content'] ?? ''), ENT_QUOTES | ENT_HTML5, 'UTF-8');
    $html    = preg_replace('#<br\s*/?>|</p>#i', "\n", $html);
    $body    = trim(html_entity_decode(strip_tags($html), ENT_QUOTES | ENT_HTML5, 'UTF-8'));
    if ($subject === '' && $body === '') {
        return;
    }
    $d = plugin_triago_score($subject, $body);
    if ($d === null) {
        return;
    }
    $gid = plugin_triago_group_id(PLUGIN_TRIAGO_GROUPS[$d['assign_to']] ?? PLUGIN_TRIAGO_GROUPS['triage']);
    if ($gid === null) {
        Toolbox::logInFile('triago', "GLPI group for '{$d['assign_to']}' not found; run the dispatcher once to create the groups\n");
        return;
    }
    // The web form sends actors as `_actors`; the API and mail collector use `_groups_id_assign`.
    if (isset($input['_actors']) && is_array($input['_actors'])) {
        $input['_actors']['assign'][] = ['itemtype' => 'Group', 'items_id' => $gid];
    } else {
        $input['_groups_id_assign'] = $gid;
    }
    $input['impact']   = PLUGIN_TRIAGO_LEVEL[$d['impact'] ?? 'high'];
    $input['urgency']  = PLUGIN_TRIAGO_LEVEL[$d['urgency'] ?? 'high'];
    $input['priority'] = PLUGIN_TRIAGO_PRIORITY[(int) $d['priority']] ?? 4;
    $input['_triago']  = $d;  // picked up by plugin_triago_item_add once the ticket has an id
}

/** After the ticket is saved: write triago's explanation as a private followup. */
function plugin_triago_item_add(Ticket $ticket)
{
    $d = $ticket->input['_triago'] ?? null;
    if (!is_array($d) || empty($d['note'])) {
        return;
    }
    $followup = new ITILFollowup();
    $followup->add([
        'itemtype'   => 'Ticket',
        'items_id'   => $ticket->getID(),
        'is_private' => 1,
        'content'    => implode('<br>', array_map('htmlspecialchars', $d['note'])),
        '_disablenotif' => true,
    ]);
}
