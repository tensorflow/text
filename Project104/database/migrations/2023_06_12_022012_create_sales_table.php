<?php

use App\Models\Country;
use App\Models\Owner;
use App\Models\Partinfo;
use App\Models\Rawmatel;
use App\Models\Initial;
use App\Models\Premob;
use App\Models\Partscap;
use App\Models\Design;
use App\Models\Dfam;
use App\Models\Itp;
use App\Models\Printing;
use App\Models\Out;
use App\Models\Post;
use App\Models\Assem;
use App\Models\Qc;
use App\Models\Finalrep;
use App\Models\Finaldate;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('sales', function (Blueprint $table) {
            $table->id();
            $table->integer('so')->unique();
            $table->string('salesman');
            $table->date('po_date');
            $table->string('customer_name');
            $table->foreignIdFor(Country::class)->constrained();
            $table->enum('type', ['A', 'B', 'C']);
            $table->enum('strategic_contract', ['Yes', 'No']);
            $table->string('paid_amount');
            $table->string('advanced_payment');
            $table->string('owner');
            $table->date('client_update_date_01');
            $table->date('so_locked_date');
            $table->date('execution_plan_discharge_date')->nullable();
            $table->date('client_update_date_02')->nullable();
            $table->date('client_kom_date')->nullable();
            $table->date('part_info_from_client_actual_date')->nullable();
            $table->integer('part_info_from_client_actual_qty')->nullable();
            $table->date('part_info_from_client_target_date')->nullable();
            $table->integer('part_info_from_client_target_qty')->nullable();
            $table->date('raw_matel_outsource_actual_date')->nullable();
            $table->integer('raw_matel_outsource_actual_qty')->nullable();
            $table->date('raw_matel_outsource_target_date')->nullable();
            $table->integer('raw_matel_outsource_target_qty')->nullable();
            $table->date('initial_assesment_actual_date')->nullable();
            $table->integer('initial_assesment_actual_qty')->nullable();
            $table->date('initial_assesment_target_date')->nullable();
            $table->integer('initial_assesment_target_qty')->nullable();
            $table->date('pre_mob_actual_date')->nullable();
            $table->integer('pre_mob_actual_qty')->nullable();
            $table->date('pre_mob_target_date')->nullable();
            $table->integer('pre_mob_target_qty')->nullable();
            $table->date('parts_capturing_actual_date')->nullable();
            $table->integer('parts_capturing_actual_qty')->nullable();
            $table->date('parts_capturing_target_date')->nullable();
            $table->integer('parts_capturing_target_qty')->nullable();
            $table->date('design_engineering_actual_date')->nullable();
            $table->integer('design_engineering_actual_qty')->nullable();
            $table->date('design_engineering_target_date')->nullable();
            $table->integer('design_engineering_target_qty')->nullable();
            $table->date('dfam_build_prep_actual_date')->nullable();
            $table->integer('dfam_build_prep_actual_qty')->nullable();
            $table->date('dfam_build_prep_target_date')->nullable();
            $table->integer('dfam_build_prep_target_qty')->nullable();
            $table->date('itp_manufacturing_document_actual_date')->nullable();
            $table->integer('itp_manufacturing_document_actual_qty')->nullable();
            $table->date('itp_manufacturing_document_target_date')->nullable();
            $table->integer('itp_manufacturing_document_target_qty')->nullable();
            $table->date('client_update_date_03')->nullable();
            $table->date('_3d_printing_actual_date')->nullable();
            $table->integer('_3d_printing_actual_qty')->nullable();
            $table->date('_3d_printing_target_date')->nullable();
            $table->integer('_3d_printing_target_qty')->nullable();
            $table->date('client_update_date_04')->nullable();
            $table->date('outsource_production_actual_date')->nullable();
            $table->integer('outsource_production_actual_qty')->nullable();
            $table->date('outsource_production_target_date')->nullable();
            $table->integer('outsource_production_target_qty')->nullable();
            $table->date('post_processing_actual_date')->nullable();
            $table->integer('post_processing_actual_qty')->nullable();
            $table->date('post_processing_target_date')->nullable();
            $table->integer('post_processing_target_qty')->nullable();
            $table->date('assembly_actual_date')->nullable();
            $table->integer('assembly_actual_qty')->nullable();
            $table->date('assembly_target_date')->nullable();
            $table->integer('assembly_target_qty')->nullable();
            $table->date('qc_testing_actual_date')->nullable();
            $table->integer('qc_testing_actual_qty')->nullable();
            $table->date('qc_testing_target_date')->nullable();
            $table->integer('qc_testing_target_qty')->nullable();
            $table->date('final_rep_estimation_data_to_customer_actual_date')->nullable();
            $table->integer('final_rep_estimation_data_to_customer_actual_qty')->nullable();
            $table->date('final_rep_estimation_data_to_customer_target_date')->nullable();
            $table->integer('final_rep_estimation_data_to_customer_target_qty')->nullable();
            $table->date('client_update_date_05')->nullable();
            $table->string('cash_collected')->nullable();
            $table->date('final_delivery_actual_date')->nullable();
            $table->integer('final_delivery_actual_qty')->nullable();
            $table->date('final_delivery_target_date')->nullable();
            $table->integer('final_delivery_target_qty')->nullable();
            $table->string('actions', 1500)->nullable();
            $table->string('lessons_learnd', 1500)->nullable();
            $table->timestamps();
            $table->softDeletes();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('sales');
    }
};
