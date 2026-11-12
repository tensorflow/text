<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class History extends Model
{
    use HasFactory;
    protected $fillable = ['sale_id', 'type', 'desciption', 'sale_data'];

    public function sale(): BelongsTo
    {
        return $this->belongsTo(Sale::class);
    }
}
