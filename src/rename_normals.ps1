$start = 1
$end   = 90

for ($i = $start; $i -le $end; $i++) {
    $oldName = "normal-$i.mp4"
    $newName = ("normal-{0:000}.mp4" -f $i)

    if (Test-Path $oldName) {
        Write-Host "Renombrando $oldName → $newName"
        Rename-Item $oldName $newName
    } else {
        Write-Host "Saltando $oldName (no existe)"
    }
}
