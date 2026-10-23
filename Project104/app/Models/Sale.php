<?php

namespace App\Models;

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Sale extends Model
{
    use HasFactory, SoftDeletes;

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'so',
        'salesman',
        'po_date',
        'customer_name',
        'country_id',
        'type',
        'strategic_contract',
        'paid_amount',
        'advanced_payment',
        'owner',
        'client_update_date_01',
        'so_locked_date',
        // Owner
        'execution_plan_discharge_date',
        'client_update_date_02',
        'client_kom_date',
        'part_info_from_client_actual_date',
        'part_info_from_client_actual_qty',
        'part_info_from_client_target_date',
        'part_info_from_client_target_qty',
        // Production
        'raw_matel_outsource_actual_date',
        'raw_matel_outsource_actual_qty',
        'raw_matel_outsource_target_date',
        'raw_matel_outsource_target_qty',
        //Engineering / Owner
        'initial_assesment_actual_date',
        'initial_assesment_actual_qty',
        'initial_assesment_target_date',
        'initial_assesment_target_qty',
        'pre_mob_actual_date',
        'pre_mob_actual_qty',
        'pre_mob_target_date',
        'pre_mob_target_qty',
        //Engineering
        'parts_capturing_actual_date',
        'parts_capturing_actual_qty',
        'parts_capturing_target_date',
        'parts_capturing_target_qty',
        'design_engineering_actual_date',
        'design_engineering_actual_qty',
        'design_engineering_target_date',
        'design_engineering_target_qty',
        'dfam_build_prep_actual_date',
        'dfam_build_prep_actual_qty',
        'dfam_build_prep_target_date',
        'dfam_build_prep_target_qty',
        //quality / Owner
        'itp_manufacturing_document_actual_date',
        'itp_manufacturing_document_actual_qty',
        'itp_manufacturing_document_target_date',
        'itp_manufacturing_document_target_qty',
        //Production Owner
        '_3d_printing_actual_date',
        '_3d_printing_actual_qty',
        '_3d_printing_target_date',
        '_3d_printing_target_qty',
        //Engineering
        'outsource_production_actual_date',
        'outsource_production_actual_qty',
        'outsource_production_target_date',
        'outsource_production_target_qty',
        'post_processing_actual_date',
        'post_processing_actual_qty',
        'post_processing_target_date',
        'post_processing_target_qty',
        'assembly_actual_date',
        'assembly_actual_qty',
        'assembly_target_date',
        'assembly_target_qty',
        //Quality
        'qc_testing_actual_date',
        'qc_testing_actual_qty',
        'qc_testing_target_date',
        'qc_testing_target_qty',
        'final_rep_estimation_data_to_customer_actual_date',
        'final_rep_estimation_data_to_customer_actual_qty',
        'final_rep_estimation_data_to_customer_target_date',
        'final_rep_estimation_data_to_customer_target_qty',
        //Owner / Finance 
        'client_update_date_05',
        'cash_collected',
        //Production
        'final_delivery_actual_date',
        'final_delivery_actual_qty',
        'final_delivery_target_date',
        'final_delivery_target_qty',
        //All Departments
        'actions',
        'lessons_learnd',
    ];

    public function saveHistory(string $type, string $description)
    {
        History::create([
            'sale_id' => $this->id,
            'type' => $type,
            'desciption' => $description,
            'sale_data' => serialize($this)
        ]);
    }

    public function restoreHistory()
    {
        $history = History::where('sale_id', $this->id)->latest()->firstOrFail();
        if ($history) {
            $historyData = unserialize($history->sale_data);
            $this->fill($historyData->toArray())->save();
            $history->delete();
            return true;
        } else {
            return false;
        }
    }


    /**
     * Get country.
     *
     * @return \Illuminate\Database\Eloquent\Relations\BelongsTo
     */
    public function country()
    {
        return $this->belongsTo(Country::class);
    }
    public function owner()
    {
        return $this->belongsTo(Owner::class);
    }
    public function partinfo()
    {
        return $this->belongsTo(Partinfo::class);
    }
    public function rawmatel()
    {
        return $this->belongsTo(Rawmatel::class);
    }
    public function initial()
    {
        return $this->belongsTo(Initial::class);
    }
    public function premob()
    {
        return $this->belongsTo(Premob::class);
    }
    public function partscap()
    {
        return $this->belongsTo(Partscap::class);
    }
    public function design()
    {
        return $this->belongsTo(Design::class);
    }
    public function dfam()
    {
        return $this->belongsTo(Dfam::class);
    }
    public function itp()
    {
        return $this->belongsTo(Itp::class);
    }
    public function printing()
    {
        return $this->belongsTo(Printing::class);
    }
    public function out()
    {
        return $this->belongsTo(Out::class);
    }
    public function post()
    {
        return $this->belongsTo(Post::class);
    }
    public function assem()
    {
        return $this->belongsTo(Assem::class);
    }
    public function qc()
    {
        return $this->belongsTo(Qc::class);
    }
    public function finalrep()
    {
        return $this->belongsTo(Finalrep::class);
    }
    public function finaldate()
    {
        return $this->belongsTo(Finaldate::class);
    }
}
