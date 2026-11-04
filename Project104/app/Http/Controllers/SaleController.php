<?php

namespace App\Http\Controllers;

use App\Models\Sale;
use App\Models\Country;
use Illuminate\Http\Request;
use DB;
use App\Http\Requests;
use App\Http\Controllers\Controller;
use App\Models\History;
use DateTime;
use Illuminate\Support\Facades\Validator;
use Illuminate\Validation\Rule;
use Illuminate\Validation\ValidationException;

class SaleController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    // public function index()
    // {
    //     return view('index', [
    //         'sales' => Sale::all(),
    //     ]);
    // }

    // function index()
    // {
    //     return view('index', [
    //                  'sales' => Sale::all(),
    //              ]);
    // }


    public function  check_is_so_unique(Request $request)
    {

        $sale =  Sale::where('so', '=', $request->so);
        if ($sale->count() > 0) {
            return 0;
        } else {
            return 1;
        }
    }
    public function last_update()
    {
        $last_update_date = '';
        $last_update_duration = '';
        $sale =  Sale::orderBy('updated_at', 'desc')->first();
        if ($sale != null) {
            $last_update_date = $sale->updated_at;

            $start_date = new DateTime($last_update_date);
            $since_start = $start_date->diff(new DateTime());

            // echo $since_start->y.' years<br>';
            // echo $since_start->m.' months<br>';
            // echo $since_start->d.' days<br>';
            // echo $since_start->h.' hours<br>';
            // echo $since_start->i.' minutes<br>';
            // echo $since_start->s.' seconds<br>';

            if ($since_start->m > 0) {
                $last_update_duration = $since_start->m . "Months ago.";
            } else if ($since_start->days > 0) {
                $last_update_duration = $since_start->days . "Days ago.";
            } else if ($since_start->h > 0) {
                $last_update_duration = $since_start->h . "Hours ago.";
            } else if ($since_start->i > 0) {
                $last_update_duration = $since_start->i . "Minutes ago.";
            } else {
                $last_update_duration = "Less than a minute ago.";
            }
        }

        return response()->json([
            'last_update_date' => date('d/m/Y', strtotime($last_update_date)),
            'last_update_duration' => $last_update_duration,
        ]);
    }
    public function index()
    {

        $last_update_date = '';
        $last_update_duration = '';
        $sale =  Sale::orderBy('updated_at', 'desc')->first();
        if ($sale != null) {
            $last_update_date = $sale->updated_at;

            $start_date = new DateTime($last_update_date);
            $since_start = $start_date->diff(new DateTime());

            // echo $since_start->y.' years<br>';
            // echo $since_start->m.' months<br>';
            // echo $since_start->d.' days<br>';
            // echo $since_start->h.' hours<br>';
            // echo $since_start->i.' minutes<br>';
            // echo $since_start->s.' seconds<br>';

            if ($since_start->m > 0) {
                $last_update_duration = $since_start->m . "Months ago.";
            } else if ($since_start->days > 0) {
                $last_update_duration = $since_start->days . "Days ago.";
            } else if ($since_start->h > 0) {
                $last_update_duration = $since_start->h . "Hours ago.";
            } else if ($since_start->i > 0) {
                $last_update_duration = $since_start->i . "Minutes ago.";
            } else {
                $last_update_duration = "Less than a minute ago.";
            }
        }
        return view('index', [
            'sales' => Sale::all(), 'countries' => Country::all(),
            'last_update_date' => $last_update_date,
            'last_update_duration' => $last_update_duration,
            'timestamp' => $sale->updated_at
        ]);
    }

    public function addNewSaleUi()
    {
        return view('adnew', ['countries' => Country::all(),]);
    }

    public function delete_sale(Request $request)
    {
        $sale = Sale::find($request->id);
        History::where('sale_id', $sale->id)->delete();
        $sale->delete();
    }
    public function undo_sale(Sale $sale)
    {
        if (auth()->user()->role == "superAdmin") {
            $result = $sale->restoreHistory();
            if ($result) {
                return response()->json([
                    "success" => "success"
                ], 200);
            } else {
                return response()->json([
                    "error" => '404'
                ], 404);
            }
        } else {
            return redirect('/');
        }
    }


    public function live_update_sale(Request $request)
    {

        // $inputs = $request->validate([
        //     'so'=>['required','max:30'],
        //     'salesman'=>['required','min:18','max:30']
        // ]);

        // $validatedData = $request->validate([
        //     'so' => 'required','max:30',
        //     'salesman' => 'required','min:15','max:30',
        // ]);

        // $validated = $request->validate([
        //     'so' => 'required',
        //     'salesman' => 'required|max:15',
        //     'po_date'=>'required',
        //     'customer_name'=>'required',
        // ]);

        $sale = Sale::find($request->id);

        switch ($request->col_name) {
            case 'so':
                $existingSale = Sale::where('so', $request->value)->first();
                if ($existingSale != null) {
                    throw ValidationException::withMessages([
                        'so' => 'Update Sale Error',
                    ]);
                }
                $sale->so = $request->value;
                break;
            case 'salesman':
                $request->validate([
                    'value' => 'required|string|max:50'
                ]);
                $sale->salesman = $request->value;
                break;
            case 'po_date':
                $sale->po_date = $request->value;
                $request->validate([
                    'value' => 'required|date'
                ]);
                break;
            case 'customer_name':
                $request->validate([
                    'value' => 'required|string|max:50'
                ]);
                $sale->customer_name = $request->value;
                break;
            case 'country_id':
                $request->validate([
                    'value' => 'required|string'
                ]);
                $sale->country_id = $request->value;
                break;
            case 'type':
                $request->validate([
                    'value' => 'required|string'
                ]);
                $sale->type = $request->value;
                break;
            case 'strategic_contract':
                $request->validate([
                    'value' => 'required|string'
                ]);
                $sale->strategic_contract = $request->value;
                break;
            case 'paid_amount':
                $request->validate([
                    'value' => 'required|string'
                ]);
                $sale->paid_amount = $request->value;
                break;
            case 'advanced_payment':
                $request->validate([
                    'value' => 'required|string'
                ]);
                $sale->advanced_payment = $request->value;
                break;
            case 'owner':
                $request->validate([
                    'value' => 'required|string|max:50'
                ]);
                $sale->owner = $request->value;
                break;
            case 'client_update_date_01':
                $request->validate([
                    'value' => 'required|date'
                ]);
                $sale->client_update_date_01 = $request->value;
                break;
            case 'so_locked_date':
                $request->validate([
                    'value' => 'required|date'
                ]);
                $sale->so_locked_date = $request->value;
                break;
            case 'execution_plan_discharge_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->execution_plan_discharge_date = $request->value;
                break;
            case 'client_update_date_02':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->client_update_date_02 = $request->value;
                break;
            case 'client_kom_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->client_kom_date = $request->value;
                break;
            case 'part_info_from_client_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->part_info_from_client_actual_date = $request->value;
                break;
            case 'part_info_from_client_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->part_info_from_client_actual_qty = $request->value;
                break;
            case 'part_info_from_client_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->part_info_from_client_target_date = $request->value;
                break;
            case 'part_info_from_client_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->part_info_from_client_target_qty = $request->value;
                break;
            case 'raw_matel_outsource_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->raw_matel_outsource_actual_date = $request->value;
                break;
            case 'raw_matel_outsource_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->raw_matel_outsource_actual_qty = $request->value;
                break;
            case 'raw_matel_outsource_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->raw_matel_outsource_target_date = $request->value;
                break;
            case 'raw_matel_outsource_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->raw_matel_outsource_target_qty = $request->value;
                break;
            case 'initial_assesment_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->initial_assesment_actual_date = $request->value;
                break;
            case 'initial_assesment_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->initial_assesment_actual_qty = $request->value;
                break;
            case 'initial_assesment_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->initial_assesment_target_date = $request->value;
                break;
            case 'initial_assesment_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->initial_assesment_target_qty = $request->value;
                break;
            case 'pre_mob_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->pre_mob_actual_date = $request->value;
                break;
            case 'pre_mob_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->pre_mob_actual_qty = $request->value;
                break;
            case 'pre_mob_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->pre_mob_target_date = $request->value;
                break;
            case 'pre_mob_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->pre_mob_target_qty = $request->value;
                break;
            case 'parts_capturing_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->parts_capturing_actual_date = $request->value;
                break;
            case 'parts_capturing_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->parts_capturing_actual_qty = $request->value;
                break;
            case 'parts_capturing_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->parts_capturing_target_date = $request->value;
                break;
            case 'parts_capturing_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->parts_capturing_target_qty = $request->value;
                break;
            case 'design_engineering_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->design_engineering_actual_date = $request->value;
                break;
            case 'design_engineering_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->design_engineering_actual_qty = $request->value;
                break;
            case 'design_engineering_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->design_engineering_target_date = $request->value;
                break;
            case 'design_engineering_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->design_engineering_target_qty = $request->value;
                break;
            case 'dfam_build_prep_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->dfam_build_prep_actual_date = $request->value;
                break;
            case 'dfam_build_prep_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->dfam_build_prep_actual_qty = $request->value;
                break;
            case 'dfam_build_prep_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->dfam_build_prep_target_date = $request->value;
                break;
            case 'dfam_build_prep_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->dfam_build_prep_target_qty = $request->value;
                break;
            case 'itp_manufacturing_document_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->itp_manufacturing_document_actual_date = $request->value;
                break;
            case 'itp_manufacturing_document_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->itp_manufacturing_document_actual_qty = $request->value;
                break;
            case 'itp_manufacturing_document_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->itp_manufacturing_document_target_date = $request->value;
                break;
            case 'itp_manufacturing_document_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->itp_manufacturing_document_target_qty = $request->value;
                break;
            case 'client_update_date_03':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->client_update_date_03 = $request->value;
                break;
            case '_3d_printing_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->{'_3d_printing_actual_date'} = $request->value;
                break;
            case '_3d_printing_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->{'_3d_printing_actual_qty'} = $request->value;
                break;
            case '_3d_printing_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->{'_3d_printing_target_date'} = $request->value;
                break;
            case '_3d_printing_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->{'_3d_printing_target_qty'} = $request->value;
                break;
            case 'client_update_date_04':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->client_update_date_04 = $request->value;
                break;
            case 'outsource_production_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->outsource_production_actual_date = $request->value;
                break;
            case 'outsource_production_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->outsource_production_actual_qty = $request->value;
                break;
            case 'outsource_production_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->outsource_production_target_date = $request->value;
                break;
            case 'outsource_production_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->outsource_production_target_qty = $request->value;
                break;
            case 'post_processing_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->post_processing_actual_date = $request->value;
                break;
            case 'post_processing_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->post_processing_actual_qty = $request->value;
                break;
            case 'post_processing_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->post_processing_target_date = $request->value;
                break;
            case 'post_processing_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->post_processing_target_qty = $request->value;
                break;
            case 'assembly_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->assembly_actual_date = $request->value;
                break;
            case 'assembly_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->assembly_actual_qty = $request->value;
                break;
            case 'assembly_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->assembly_target_date = $request->value;
                break;
            case 'assembly_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->assembly_target_qty = $request->value;
                break;
            case 'qc_testing_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->qc_testing_actual_date = $request->value;
                break;
            case 'qc_testing_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->qc_testing_actual_qty = $request->value;
                break;
            case 'qc_testing_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->qc_testing_target_date = $request->value;
                break;
            case 'qc_testing_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->qc_testing_target_qty = $request->value;
                break;
            case 'final_rep_estimation_data_to_customer_actual_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->final_rep_estimation_data_to_customer_actual_date = $request->value;
                break;
            case 'final_rep_estimation_data_to_customer_actual_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->final_rep_estimation_data_to_customer_actual_qty = $request->value;
                break;
            case 'final_rep_estimation_data_to_customer_target_date':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->final_rep_estimation_data_to_customer_target_date = $request->value;
                break;
            case 'final_rep_estimation_data_to_customer_target_qty':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->final_rep_estimation_data_to_customer_target_qty = $request->value;
                break;
            case 'client_update_date_05':
                $request->validate([
                    'value' => 'nullable|date'
                ]);
                $sale->client_update_date_05 = $request->value;
                break;
            case 'cash_collected':
                $request->validate([
                    'value' => 'nullable|numeric|digits_between:1,11'
                ]);
                $sale->cash_collected = $request->value;
                break;
            case 'final_delivery_actual_date':
                $request->validate([
                    'value' => 'required|date'
                ]);
                $sale->final_delivery_actual_date = $request->value;
                break;
            case 'final_delivery_actual_qty':
                $request->validate([
                    'value' => 'required|numeric|digits_between:1,11'
                ]);
                $sale->final_delivery_actual_qty = $request->value;
                break;
            case 'final_delivery_target_date':
                $request->validate([
                    'value' => 'required|date'
                ]);
                $sale->final_delivery_target_date = $request->value;
                break;
            case 'final_delivery_target_qty':
                $request->validate([
                    'value' => 'required|numeric|digits_between:1,11'
                ]);
                $sale->final_delivery_target_qty = $request->value;
                break;
            case 'actions':
                $request->validate([
                    'value' => 'nullable|string|max:1500'
                ]);
                $sale->actions = $request->value;
                break;
            case 'lessons_learnd':
                $request->validate([
                    'value' => 'nullable|string|max:1500'
                ]);
                $sale->lessons_learnd = $request->value;
                break;
        }

        $sale->save();
        $sale->saveHistory('UPDATE', 'Sale Update field ' . $request->col_name . ',Sale Id-> ' . $sale->so);
    }




    public function save_new_sale(Request $request)
    {

        $request->validate([
            // Salse
            'so' => 'required|numeric|digits_between:1,11|unique:sales,so',
            'salesman' => 'required|string|max:50',
            'po_date' => 'required|date',
            'customer_name' => 'required|string|max:50',
            'country_id' => 'required|numeric|digits_between:1,20',
            'type' => 'required|string',
            'strategic_contract' => 'required|string',
            'paid_amount' => 'required|string|max:191',
            'advanced_payment' => 'required|string|max:191',
            'owner' => 'required|string|max:50',
            'client_update_date_01' => 'required|date',
            'so_locked_date' => 'required|date',
            // Owner
            'execution_plan_discharge_date' => 'nullable|date',
            'client_update_date_02' => 'nullable|date',
            'client_kom_date' => 'nullable|date',
            'part_info_from_client_actual_date' => 'nullable|date',
            'part_info_from_client_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'part_info_from_client_target_date' => 'nullable|date',
            'part_info_from_client_target_qty' => 'nullable|numeric|digits_between:1,11',
            // Production
            'raw_matel_outsource_actual_date' => 'nullable|date',
            'raw_matel_outsource_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'raw_matel_outsource_target_date' => 'nullable|date',
            'raw_matel_outsource_target_qty' => 'nullable|numeric|digits_between:1,11',
            //Engineering / Owner
            'initial_assesment_actual_date' => 'nullable|date',
            'initial_assesment_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'initial_assesment_target_date' => 'nullable|date',
            'initial_assesment_target_qty' => 'nullable|numeric|digits_between:1,11',
            'pre_mob_actual_date' => 'nullable|date',
            'pre_mob_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'pre_mob_target_date' => 'nullable|date',
            'pre_mob_target_qty' => 'nullable|numeric|digits_between:1,11',
            //Engineering
            'parts_capturing_actual_date' => 'nullable|date',
            'parts_capturing_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'parts_capturing_target_date' => 'nullable|date',
            'parts_capturing_target_qty' => 'nullable|numeric|digits_between:1,11',
            'design_engineering_actual_date' => 'nullable|date',
            'design_engineering_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'design_engineering_target_date' => 'nullable|date',
            'design_engineering_target_qty' => 'nullable|numeric|digits_between:1,11',
            'dfam_build_prep_actual_date' => 'nullable|date',
            'dfam_build_prep_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'dfam_build_prep_target_date' => 'nullable|date',
            'dfam_build_prep_target_qty' => 'nullable|numeric|digits_between:1,11',
            //quality / Owner
            'itp_manufacturing_document_actual_date' => 'nullable|date',
            'itp_manufacturing_document_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'itp_manufacturing_document_target_date' => 'nullable|date',
            'itp_manufacturing_document_target_qty' => 'nullable|numeric|digits_between:1,11',
            //Production Owner
            '_3d_printing_actual_date' => 'nullable|date',
            '_3d_printing_actual_qty' => 'nullable|numeric|digits_between:1,11',
            '_3d_printing_target_date' => 'nullable|date',
            '_3d_printing_target_qty' => 'nullable|numeric|digits_between:1,11',
            //Engineering
            'outsource_production_actual_date' => 'nullable|date',
            'outsource_production_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'outsource_production_target_date' => 'nullable|date',
            'outsource_production_target_qty' => 'nullable|numeric|digits_between:1,11',
            'post_processing_actual_date' => 'nullable|date',
            'post_processing_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'post_processing_target_date' => 'nullable|date',
            'post_processing_target_qty' => 'nullable|numeric|digits_between:1,11',
            'assembly_actual_date' => 'nullable|date',
            'assembly_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'assembly_target_date' => 'nullable|date',
            'assembly_target_qty' => 'nullable|numeric|digits_between:1,11',
            //Quality
            'qc_testing_actual_date' => 'nullable|date',
            'qc_testing_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'qc_testing_target_date' => 'nullable|date',
            'qc_testing_target_qty' => 'nullable|numeric|digits_between:1,11',
            'final_rep_estimation_data_to_customer_actual_date' => 'nullable|date',
            'final_rep_estimation_data_to_customer_actual_qty' => 'nullable|numeric|digits_between:1,11',
            'final_rep_estimation_data_to_customer_target_date' => 'nullable|date',
            'final_rep_estimation_data_to_customer_target_qty' => 'nullable|numeric|digits_between:1,11',
            //Owner / Finance 
            'client_update_date_05' => 'nullable|date',
            'cash_collected' => 'nullable|string|max:191',
            //Production
            'final_delivery_actual_date' => 'required|date',
            'final_delivery_actual_qty' => 'required|numeric|digits_between:1,11',
            'final_delivery_target_date' => 'required|date',
            'final_delivery_target_qty' => 'required|numeric|digits_between:1,11',
            //All Departments
            'actions' => 'nullable|string|max:1500',
            'lessons_learnd' => 'nullable|string|max:1500'


        ]);

        $sale = new Sale();
        // Assign the request values to the corresponding columns

        /** Salse */
        $sale->so = $request->so;
        $sale->salesman = $request->salesman;
        $sale->po_date = $request->po_date;
        $sale->customer_name = $request->customer_name;
        $sale->country_id = $request->country_id;
        $sale->type = $request->type;
        $sale->strategic_contract = $request->strategic_contract;
        $sale->paid_amount = $request->paid_amount;
        $sale->advanced_payment = $request->advanced_payment;
        $sale->owner = $request->owner;
        $sale->client_update_date_01 = $request->client_update_date_01;
        $sale->so_locked_date = $request->so_locked_date;

        /**  Owner */
        $sale->execution_plan_discharge_date = $request->execution_plan_discharge_date;
        $sale->client_update_date_02 = $request->client_update_date_02;
        $sale->client_kom_date = $request->client_kom_date;
        $sale->part_info_from_client_actual_date = $request->part_info_from_client_actual_date;
        $sale->part_info_from_client_actual_qty = $request->part_info_from_client_actual_qty;
        $sale->part_info_from_client_target_date = $request->part_info_from_client_target_date;
        $sale->part_info_from_client_target_qty = $request->part_info_from_client_target_qty;

        /** Production */
        $sale->raw_matel_outsource_actual_date = $request->raw_matel_outsource_actual_date;
        $sale->raw_matel_outsource_actual_qty = $request->raw_matel_outsource_actual_qty;
        $sale->raw_matel_outsource_target_date = $request->raw_matel_outsource_target_date;
        $sale->raw_matel_outsource_target_qty = $request->raw_matel_outsource_target_qty;

        /** Engineering / Owner */
        $sale->initial_assesment_actual_date = $request->initial_assesment_actual_date;
        $sale->initial_assesment_actual_qty = $request->initial_assesment_actual_qty;
        $sale->initial_assesment_target_date = $request->initial_assesment_target_date;
        $sale->initial_assesment_target_qty = $request->initial_assesment_target_qty;
        $sale->pre_mob_actual_date = $request->pre_mob_actual_date;
        $sale->pre_mob_actual_qty = $request->pre_mob_actual_qty;
        $sale->pre_mob_target_date = $request->pre_mob_target_date;
        $sale->pre_mob_target_qty = $request->pre_mob_target_qty;

        /** Engineering */
        $sale->parts_capturing_actual_date = $request->parts_capturing_actual_date;
        $sale->parts_capturing_actual_qty = $request->parts_capturing_actual_qty;
        $sale->parts_capturing_target_date = $request->parts_capturing_target_date;
        $sale->parts_capturing_target_qty = $request->parts_capturing_target_qty;
        $sale->design_engineering_actual_date = $request->design_engineering_actual_date;
        $sale->design_engineering_actual_qty = $request->design_engineering_actual_qty;
        $sale->design_engineering_target_date = $request->design_engineering_target_date;
        $sale->design_engineering_target_qty = $request->design_engineering_target_qty;
        $sale->dfam_build_prep_actual_date = $request->dfam_build_prep_actual_date;
        $sale->dfam_build_prep_actual_qty = $request->dfam_build_prep_actual_qty;
        $sale->dfam_build_prep_target_date = $request->dfam_build_prep_target_date;
        $sale->dfam_build_prep_target_qty = $request->dfam_build_prep_target_qty;

        /** quality / Owner */
        $sale->itp_manufacturing_document_actual_date = $request->itp_manufacturing_document_actual_date;
        $sale->itp_manufacturing_document_actual_qty = $request->itp_manufacturing_document_actual_qty;
        $sale->itp_manufacturing_document_target_date = $request->itp_manufacturing_document_target_date;
        $sale->itp_manufacturing_document_target_qty = $request->itp_manufacturing_document_target_qty;
        $sale->client_update_date_03 = $request->client_update_date_03;

        /** Production Owner  */
        $sale->{'_3d_printing_actual_date'} = $request->{'_3d_printing_actual_date'};
        $sale->{'_3d_printing_actual_qty'} = $request->{'_3d_printing_actual_qty'};
        $sale->{'_3d_printing_target_date'} = $request->{'_3d_printing_target_date'};
        $sale->{'_3d_printing_target_qty'} = $request->{'_3d_printing_target_qty'};
        $sale->client_update_date_04 = $request->client_update_date_04;

        /** Engineering */
        $sale->outsource_production_actual_date = $request->outsource_production_actual_date;
        $sale->outsource_production_actual_qty = $request->outsource_production_actual_qty;
        $sale->outsource_production_target_date = $request->outsource_production_target_date;
        $sale->outsource_production_target_qty = $request->outsource_production_target_qty;
        $sale->post_processing_actual_date = $request->post_processing_actual_date;
        $sale->post_processing_actual_qty = $request->post_processing_actual_qty;
        $sale->post_processing_target_date = $request->post_processing_target_date;
        $sale->post_processing_target_qty = $request->post_processing_target_qty;
        $sale->assembly_actual_date = $request->assembly_actual_date;
        $sale->assembly_actual_qty = $request->assembly_actual_qty;
        $sale->assembly_target_date = $request->assembly_target_date;
        $sale->assembly_target_qty = $request->assembly_target_qty;

        /** Quality */
        $sale->qc_testing_actual_date = $request->qc_testing_actual_date;
        $sale->qc_testing_actual_qty = $request->qc_testing_actual_qty;
        $sale->qc_testing_target_date = $request->qc_testing_target_date;
        $sale->qc_testing_target_qty = $request->qc_testing_target_qty;
        $sale->final_rep_estimation_data_to_customer_actual_date = $request->final_rep_estimation_data_to_customer_actual_date;
        $sale->final_rep_estimation_data_to_customer_actual_qty = $request->final_rep_estimation_data_to_customer_actual_qty;
        $sale->final_rep_estimation_data_to_customer_target_date = $request->final_rep_estimation_data_to_customer_target_date;
        $sale->final_rep_estimation_data_to_customer_target_qty = $request->final_rep_estimation_data_to_customer_target_qty;

        /** Owner / Finance */
        $sale->client_update_date_05 = $request->client_update_date_05;
        $sale->cash_collected = $request->cash_collected;

        /** Production */
        $sale->final_delivery_actual_date = $request->final_delivery_actual_date;
        $sale->final_delivery_actual_qty = $request->final_delivery_actual_qty;
        $sale->final_delivery_target_date = $request->final_delivery_target_date;
        $sale->final_delivery_target_qty = $request->final_delivery_target_qty;

        /** All Departments */
        $sale->actions = $request->actions;
        $sale->lessons_learnd = $request->lessons_learnd;

        // Save the model to the database
        $sale->save();
        $sale->saveHistory('CREATE', 'Sale created id->' . $sale->so);
        return redirect('/');
    }

    public function trash()
    {
        if (auth()->user()->role == "superAdmin") {
            return view('trash', [
                'sales' => Sale::onlyTrashed()->orderBy('deleted_at', 'desc')->get()
            ]);
        } else {
            return redirect('/');
        }
    }
    public function restore_sale(Request $request)
    {
        Sale::withTrashed()->find($request->id)->restore();
    }
}
