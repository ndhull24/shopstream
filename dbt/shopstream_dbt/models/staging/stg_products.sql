with source as (
    select * from raw_products
),

cleaned as (
    select
        product_id,
        product_name,
        category,
        price,
        cost,
        round((price - cost) / price, 4)    as margin,
        stock_qty,
        cast(is_active as boolean)           as is_active,
        case
            when price < 20   then 'budget'
            when price < 100  then 'mid'
            when price < 300  then 'premium'
            else 'luxury'
        end                                  as price_tier
    from source
    where product_id is not null
)

select * from cleaned